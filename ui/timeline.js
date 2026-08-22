// Timeline: an Audacity-style multi-track view over a growing recording.
//
// The single source of truth for "what is on screen" is the viewport
// {s0, spp}: left edge in samples and samples per device pixel. The ruler,
// every track, the playhead, and all mouse math derive from it, so tracks
// can never drift out of alignment. Track *content* is delegated to a
// renderTrack callback; the Timeline decides *when* and *what region* to
// render (incremental columns while following a recording, full repaints on
// zoom/pan), and handles DPR, resize, zoom, pan, follow mode, and seeking.

const MIN_SPP = 16;          // max zoom in: ~0.3ms per px at 48kHz
const HEAD_W = 110;          // css px, track header column

export class Timeline {
    /**
     * opts: {
     *   tracks: [{id, label, views: [name...], view}],
     *   renderTrack(track, canvas, vp, dirtyX0),  // vp: {s0, spp, w, h, dpr}
     *   onSeek(sampleIdx),
     *   onViewChange(track),
     *   onTrackDrag(track, pos, phase)  // optional; pos: {x, y, sample, w, h}
     *                                   // return true to consume (blocks seek)
     * }
     */
    constructor(root, opts) {
        this.opts = opts;
        this.root = root;
        this.sampleRate = 48000;
        this.total = 0;          // recording length in samples
        this.s0 = 0;             // viewport left edge (samples)
        this.spp = null;         // samples per device px; null = fit
        this.followMode = false;
        this.playhead = null;    // sample idx or null
        this.dpr = window.devicePixelRatio || 1;
        this.tracks = opts.tracks.map(t => ({
            ...t,
            canvas: null,
            rendered: null,      // {s0, spp, end, view} of current canvas content
        }));

        this.#buildDom();
        this.#bindInteractions();
        new ResizeObserver(() => this.#resize()).observe(this.rulerCanvas.parentElement);
        this.#resize();
    }

    #buildDom() {
        const el = (tag, cls, parent) => {
            const e = document.createElement(tag);
            if (cls) e.className = cls;
            parent?.appendChild(e);
            return e;
        };
        this.root.classList.add('tl');

        const rulerRow = el('div', 'tl-row', this.root);
        const corner = el('div', 'tl-corner', rulerRow);
        for (const [txt, title, fn] of [
            ['−', 'Zoom out', () => this.zoomBy(2)],
            ['+', 'Zoom in', () => this.zoomBy(0.5)],
            ['Fit', 'Zoom to whole recording', () => this.fit()],
        ]) {
            const b = el('button', 'tl-zoom-btn', corner);
            b.textContent = txt;
            b.title = title;
            b.addEventListener('click', fn);
        }
        const rulerWrap = el('div', 'tl-canvas-wrap tl-ruler-wrap', rulerRow);
        this.rulerCanvas = el('canvas', 'tl-ruler', rulerWrap);

        this.bodyEl = el('div', 'tl-body', this.root);
        for (const t of this.tracks) {
            const row = el('div', 'tl-row tl-track', this.bodyEl);
            const head = el('div', 'tl-head', row);
            el('div', 'tl-label', head).textContent = t.label;
            const sel = el('select', 'tl-view-select', head);
            for (const v of t.views) {
                const o = el('option', null, sel);
                o.value = v;
                o.textContent = v[0].toUpperCase() + v.slice(1);
            }
            sel.value = t.view;
            sel.addEventListener('change', () => {
                t.view = sel.value;
                this.invalidate(t.id);
                this.opts.onViewChange?.(t);
            });
            t.viewSelect = sel;
            const wrap = el('div', 'tl-canvas-wrap', row);
            t.canvas = el('canvas', 'tl-track-canvas', wrap);
            t.canvas.dataset.trackId = t.id;
        }

        // Playhead: a positioned div over the canvas column — no canvas
        // layers, no snapshot/restore, alignment is shared by construction.
        this.playheadEl = el('div', 'tl-playhead', this.bodyEl);
        this.playheadEl.style.display = 'none';
    }

    // --- Geometry ---

    /** Device-px width of the track canvases. */
    get w() {
        return this.tracks[0].canvas.width;
    }

    #cssWidth() {
        return this.tracks[0].canvas.parentElement.clientWidth;
    }

    #resize() {
        this.dpr = window.devicePixelRatio || 1;
        const cssW = this.#cssWidth();
        if (cssW === 0) return;
        for (const t of this.tracks) {
            const h = t.canvas.parentElement.clientHeight;
            t.canvas.width = Math.round(cssW * this.dpr);
            t.canvas.height = Math.round(h * this.dpr);
            t.rendered = null;
        }
        this.rulerCanvas.width = Math.round(cssW * this.dpr);
        this.rulerCanvas.height = Math.round(this.rulerCanvas.parentElement.clientHeight * this.dpr);
        this.renderedRuler = null;
        this.render();
    }

    /** Effective samples-per-pixel (resolves fit mode). */
    #effSpp() {
        if (this.spp !== null) return this.spp;
        return Math.max(MIN_SPP, this.total / Math.max(1, this.w));
    }

    sampleAtX(cssX) {
        return this.s0 + cssX * this.dpr * this.#effSpp();
    }

    xAtSample(s) {
        return (s - this.s0) / this.#effSpp() / this.dpr; // css px
    }

    // --- Viewport control ---

    setSampleRate(sr) {
        this.sampleRate = sr;
        this.renderedRuler = null;
    }

    reset() {
        this.total = 0;
        this.s0 = 0;
        this.spp = null;
        this.playhead = null;
        this.invalidate();
    }

    setTotal(samples) {
        this.total = samples;
    }

    /** Follow mode: keep the right edge pinned to the end of the recording. */
    follow(on, windowSeconds = 10) {
        this.followMode = on;
        if (on) this.spp = (windowSeconds * this.sampleRate) / Math.max(1, this.w);
    }

    zoomBy(factor, atCssX = null) {
        const spp = this.#effSpp();
        const maxSpp = Math.max(this.total / Math.max(1, this.w), MIN_SPP);
        const newSpp = Math.min(Math.max(spp * factor, MIN_SPP), Math.max(maxSpp, spp));
        const anchor = atCssX !== null
            ? this.sampleAtX(atCssX)
            : this.s0 + (this.w / 2) * spp;
        const anchorPx = atCssX !== null ? atCssX * this.dpr : this.w / 2;
        this.spp = newSpp;
        this.s0 = Math.max(0, anchor - anchorPx * newSpp);
        this.followMode = false;
        this.render();
    }

    fit() {
        this.spp = null;
        this.s0 = 0;
        this.followMode = false;
        this.render();
    }

    panBy(cssDx) {
        const spp = this.#effSpp();
        const maxS0 = Math.max(0, this.total - this.w * spp);
        this.s0 = Math.min(Math.max(0, this.s0 + cssDx * this.dpr * spp), maxS0);
        this.spp = spp; // pin (leave fit mode)
        this.followMode = false;
        this.render();
    }

    setPlayhead(sampleIdx) {
        this.playhead = sampleIdx;
        this.#positionPlayhead();
    }

    /** Set how far a track's data actually extends (defaults to the total).
     *  Rolls the repaint watermark back if the data shrank (playback
     *  re-processing truncates the output at the seek position). */
    setDataEnd(trackId, sampleIdx) {
        const t = this.tracks.find(t => t.id === trackId);
        if (!t) return;
        t.dataEnd = sampleIdx;
        if (t.rendered) t.rendered.end = Math.min(t.rendered.end, sampleIdx);
    }

    /** Force a full repaint of one track (or all). */
    invalidate(trackId = null) {
        for (const t of this.tracks) {
            if (trackId === null || t.id === trackId) t.rendered = null;
        }
        this.render();
    }

    getTrack(id) {
        return this.tracks.find(t => t.id === id);
    }

    // --- Rendering ---

    /** Render everything that is out of date. Cheap when nothing changed. */
    render() {
        if (this.w === 0) return;
        const spp = this.#effSpp();
        if (this.followMode) {
            // Pin right edge to the end; quantize s0 to a whole column so
            // incremental scroll-blits move by integer pixels.
            const s0 = Math.max(0, this.total - this.w * spp);
            this.s0 = Math.floor(s0 / spp) * spp;
        }
        let more = false;
        for (const t of this.tracks) more = this.#renderTrack(t, spp) || more;
        this.#renderRuler(spp);
        this.#positionPlayhead();
        // A repaint hit the per-frame budget: continue next frame so big
        // repaints (zoom, fit) finish without an external render driver.
        if (more && !this.#continuation) {
            this.#continuation = requestAnimationFrame(() => {
                this.#continuation = null;
                this.render();
            });
        }
    }

    #continuation = null;

    #renderTrack(t, spp) {
        const vp = { s0: this.s0, spp, w: t.canvas.width, h: t.canvas.height, dpr: this.dpr };
        const r = t.rendered;
        // The repaint watermark stops at the track's real data end (the
        // output lags the input by the pipeline latency); columns painted
        // before their data existed get repainted when it arrives.
        const dataEnd = Math.min(t.dataEnd ?? this.total, this.total);
        const end = Math.min(dataEnd, this.s0 + vp.w * spp);

        // Budget per frame: spectrogram columns cost an FFT each, so a full
        // repaint of a long recording is spread over a few frames (the
        // watermark resumes where the previous frame stopped) instead of
        // blocking the main thread.
        const MAX_COLS = 128;

        let x0, x1;
        if (!r || r.spp !== spp || r.view !== t.view) {
            // Zoom/view change: background-fill now (cheap), then repaint
            // data columns progressively from the left.
            const ctx = t.canvas.getContext('2d');
            ctx.fillStyle = 'rgb(10,10,20)'; // matches the views' background
            ctx.fillRect(0, 0, vp.w, vp.h);
            x0 = 0;
            x1 = Math.max(0, Math.min(vp.w, Math.ceil((end - this.s0) / spp)));
        } else if (r.s0 === this.s0 && r.end >= end) {
            return false; // up to date
        } else {
            const dxPx = Math.round((this.s0 - r.s0) / spp);
            if (dxPx !== 0 && Math.abs(dxPx) < vp.w) {
                // Scroll: self-blit the overlap, then repaint the gap.
                const ctx = t.canvas.getContext('2d');
                ctx.drawImage(t.canvas, -dxPx, 0);
            }
            if (Math.abs(dxPx) >= vp.w || dxPx < 0) {
                // Jumped or scrolled left: repaint everything (user pan).
                x0 = 0;
                x1 = vp.w;
            } else {
                // Repaint from the old data watermark to the new data end;
                // when we scrolled, the scrolled-in gap [w-dx, w) must be
                // painted too (it usually overlaps the watermark region).
                const watermarkX = Math.floor((r.end - this.s0) / spp);
                const dataEndX = Math.ceil((end - this.s0) / spp);
                x0 = Math.max(0, Math.min(vp.w - dxPx, watermarkX));
                x1 = dxPx > 0 ? vp.w : Math.max(0, Math.min(vp.w, dataEndX));
            }
        }
        const truncated = x1 > x0 + MAX_COLS;
        x1 = Math.min(x1, x0 + MAX_COLS);
        if (x0 < x1) this.opts.renderTrack(t, t.canvas, vp, x0, x1);
        // Watermark reflects what was actually rendered so the next frame
        // resumes from there.
        t.rendered = {
            s0: this.s0,
            spp,
            end: Math.min(end, this.s0 + x1 * spp),
            view: t.view,
        };
        return truncated;
    }

    #renderRuler(spp) {
        const c = this.rulerCanvas;
        const key = `${this.s0}:${spp}:${c.width}`;
        if (this.renderedRuler === key) return;
        this.renderedRuler = key;

        const ctx = c.getContext('2d');
        const { width: w, height: h } = c;
        ctx.fillStyle = '#10101c';
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.fillStyle = 'rgba(255,255,255,0.75)';
        ctx.font = `${10 * this.dpr}px monospace`;

        // Nice tick step: 1/2/5 × 10^k seconds, aiming for ~90px major ticks.
        const secPerPx = spp / this.sampleRate;
        const target = secPerPx * 90 * this.dpr;
        const pow = 10 ** Math.floor(Math.log10(target));
        const step = [1, 2, 5, 10].map(m => m * pow).find(s => s >= target) || 10 * pow;

        const t0 = (this.s0 / this.sampleRate);
        const t1 = t0 + (w * spp) / this.sampleRate;
        ctx.beginPath();
        for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
            const x = (t * this.sampleRate - this.s0) / spp;
            ctx.moveTo(x, h * 0.4);
            ctx.lineTo(x, h);
            const label = step >= 1 ? `${Math.round(t)}s` : `${t.toFixed(step >= 0.1 ? 1 : 2)}s`;
            ctx.fillText(label, x + 3 * this.dpr, h * 0.55);
        }
        // Minor ticks.
        const minor = step / 5;
        for (let t = Math.ceil(t0 / minor) * minor; t <= t1; t += minor) {
            const x = (t * this.sampleRate - this.s0) / spp;
            ctx.moveTo(x, h * 0.75);
            ctx.lineTo(x, h);
        }
        ctx.stroke();
    }

    #positionPlayhead() {
        if (this.playhead === null) {
            this.playheadEl.style.display = 'none';
            return;
        }
        const x = this.xAtSample(this.playhead);
        const cssW = this.#cssWidth();
        if (x < 0 || x > cssW) {
            this.playheadEl.style.display = 'none';
            return;
        }
        this.playheadEl.style.display = '';
        this.playheadEl.style.left = `${HEAD_W + x}px`;
    }

    // --- Interactions ---

    #bindInteractions() {
        const surfaces = [this.rulerCanvas, ...this.tracks.map(t => t.canvas)];
        for (const c of surfaces) {
            c.addEventListener('wheel', e => {
                e.preventDefault();
                const rect = c.getBoundingClientRect();
                const x = e.clientX - rect.left;
                if (e.shiftKey) {
                    this.panBy(e.deltaY);
                } else {
                    this.zoomBy(Math.exp(e.deltaY * 0.002), x);
                }
            }, { passive: false });
        }

        // Click = seek; drag on a track = delegated (contour editing).
        for (const t of this.tracks) {
            t.canvas.addEventListener('pointerdown', e => {
                const pos = this.#trackPos(t, e);
                if (this.opts.onTrackDrag?.(t, pos, 'start')) {
                    t.canvas.setPointerCapture(e.pointerId);
                    t.dragging = true;
                } else {
                    t.pendingSeek = pos.sample;
                }
            });
            t.canvas.addEventListener('pointermove', e => {
                if (t.dragging) this.opts.onTrackDrag(t, this.#trackPos(t, e), 'move');
            });
            t.canvas.addEventListener('pointerup', e => {
                if (t.dragging) {
                    this.opts.onTrackDrag(t, this.#trackPos(t, e), 'end');
                    t.dragging = false;
                } else if (t.pendingSeek !== undefined) {
                    this.opts.onSeek?.(Math.max(0, Math.min(t.pendingSeek, this.total)));
                    t.pendingSeek = undefined;
                }
            });
        }
        this.rulerCanvas.addEventListener('click', e => {
            const rect = this.rulerCanvas.getBoundingClientRect();
            const s = this.sampleAtX(e.clientX - rect.left);
            this.opts.onSeek?.(Math.max(0, Math.min(s, this.total)));
        });
    }

    #trackPos(t, e) {
        const rect = t.canvas.getBoundingClientRect();
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        return {
            x: cssX * this.dpr,
            y: cssY * this.dpr,
            sample: this.sampleAtX(cssX),
            w: t.canvas.width,
            h: t.canvas.height,
        };
    }
}
