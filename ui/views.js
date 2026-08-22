// Track content renderers built on one shared series-plotting backend.
//
// Pitch and waveform are both "points over time" (one per YIN hop, one per
// audio sample); they differ only in point rate and y mapping. The backend
// is zoom-adaptive with a continuous transition:
//
//   - points closer than ~2 css px: per-column min/max envelope — the exact
//     shape squashed points would form, cheap at any density
//   - wider: a thin polyline through the points; dots ramp up from
//     line-width size to full radius as spacing grows, so the line
//     "grows points" naturally while zooming in
//
// (Spectrogram pixels come from Rust via draw_*_spectrogram_range.)

const DENSE_LIMIT_CSS = 2; // point spacing below which we draw the envelope
const DOT_RAMP_START = 3;  // css px spacing where dots begin to grow
const DOT_RAMP_FULL = 9;   // css px spacing of full-size dots
const DOT_R_CSS = 1.6;     // full dot radius (the classic pitch-dot look)
const LINE_W_CSS = 1.1;

const BG = 'rgb(10,10,20)';

// --- Shared backend ---

/** Vertical min/max strokes, one per device-px column: the dense regime. */
function drawEnvelopeColumns(ctx, cols, dpr, color) {
    ctx.fillStyle = color;
    const lw = LINE_W_CSS * dpr;
    for (const c of cols) {
        ctx.fillRect(c.x, c.y0 - lw / 2, 1, c.y1 - c.y0 + lw);
    }
}

/**
 * Polyline through points with zoom-ramped dots: the sparse regime.
 * pts: array of {x, y} or null (gap breaks the line).
 */
function drawLinesAndDots(ctx, pts, spacingCss, dpr, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = LINE_W_CSS * dpr;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let pen = false;
    for (const p of pts) {
        if (!p) { pen = false; continue; }
        if (pen) ctx.lineTo(p.x, p.y);
        else ctx.moveTo(p.x, p.y);
        pen = true;
    }
    ctx.stroke();

    // Dots grow from line-width size (invisible against the line, but keeps
    // isolated points visible) to full radius as the points spread out.
    const ramp = Math.min(1, Math.max(0, (spacingCss - DOT_RAMP_START) / (DOT_RAMP_FULL - DOT_RAMP_START)));
    const r = Math.max(ramp * DOT_R_CSS, LINE_W_CSS * 0.6) * dpr;
    ctx.fillStyle = color;
    ctx.beginPath();
    for (const p of pts) {
        if (!p) continue;
        ctx.moveTo(p.x + r, p.y);
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    }
    ctx.fill();
}

// --- Pitch axis: dynamic log-frequency scale shared by all pitch views ---

// Work in MIDI semitones (69 = A4 = 440Hz): linear in log-frequency and
// directly usable for the note grid.
const freqToMidi = f => 69 + 12 * Math.log2(f / 440);
const midiToFreq = m => 440 * 2 ** ((m - 69) / 12);

const SCALE_DEFAULT_LO = 36;  // C2
const SCALE_DEFAULT_HI = 84;  // C6
const SCALE_PAD_ST = 2;       // buffer on each side of the data
const SCALE_MIN_SPAN_ST = 12; // never tighter than 1 octave
const SCALE_SHRINK_SLACK = 4; // shrink only when this much slack per side
const SCALE_ABS_LO = 12;      // C0
const SCALE_ABS_HI = 108;     // C8

/**
 * The y axis for pitch views. One instance is shared by the input and
 * output tracks (and the contour editor) so they can never disagree.
 *
 * update() fits the range to the 5th..95th percentile of the voiced hops
 * (outlier rejection), padded and clamped, and applies it with hysteresis:
 * expand as soon as padded data crosses the current bounds, shrink only
 * when there is generous slack — so the axis is not busy while singing.
 */
export class PitchScale {
    constructor() {
        this.lo = SCALE_DEFAULT_LO;
        this.hi = SCALE_DEFAULT_HI;
    }

    /** freqLists: iterable of Float32Arrays (Hz per hop, 0 = unvoiced).
     *  Returns true if the applied range changed (views need a repaint). */
    update(freqLists) {
        const st = [];
        for (const list of freqLists) {
            for (const f of list) if (f > 0) st.push(freqToMidi(f));
        }
        if (st.length < 8) return false; // not enough data to trust
        st.sort((a, b) => a - b);
        const p5 = st[Math.floor(st.length * 0.05)];
        const p95 = st[Math.min(st.length - 1, Math.floor(st.length * 0.95))];

        let lo = p5 - SCALE_PAD_ST;
        let hi = p95 + SCALE_PAD_ST;
        const deficit = SCALE_MIN_SPAN_ST - (hi - lo);
        if (deficit > 0) {
            lo -= deficit / 2;
            hi += deficit / 2;
        }
        lo = Math.max(SCALE_ABS_LO, Math.floor(lo));
        hi = Math.min(SCALE_ABS_HI, Math.ceil(hi));

        const mustExpand = lo < this.lo || hi > this.hi;
        const canShrink = lo > this.lo + SCALE_SHRINK_SLACK || hi < this.hi - SCALE_SHRINK_SLACK;
        if (!mustExpand && !canShrink) return false;
        // Take the fresh fit entirely (it already includes the pad);
        // unioning with the old range would only ever creep wider.
        this.lo = lo;
        this.hi = hi;
        return true;
    }

    /** Frequency (Hz) -> y device px. Top = high pitch. -1 for f<=0. */
    freqToY(freq, height) {
        if (freq <= 0) return -1;
        const t = (freqToMidi(freq) - this.lo) / (this.hi - this.lo);
        return (1 - t) * height;
    }

    /** y device px -> frequency (Hz). */
    yToFreq(y, height) {
        const t = 1 - y / height;
        return midiToFreq(this.lo + t * (this.hi - this.lo));
    }
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/**
 * Background + note grid + labels for a pitch track.
 *
 * Lines and labels adapt to both the axis density and the active musical
 * scale (opts.noteBits, bit per pitch class; opts.root):
 * - in-scale notes get brighter lines, the root brightest; out-of-scale
 *   lines fade (and drop entirely when the grid gets dense)
 * - labels sit vertically centered on their line, and their density follows
 *   px-per-semitone: every labeled-scale note when roomy, root + octave C's
 *   in between, octave C's only when tight
 */
export function drawPitchGrid(ctx, w, h, dpr, scale, opts = {}) {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    const { noteBits = 0, root = 0 } = opts;
    // "Off" and full chromatic carry no note preference.
    const hasScale = noteBits !== 0 && noteBits !== 0xFFF;
    const semitonePx = h / (scale.hi - scale.lo);

    ctx.font = `${10 * dpr}px monospace`;
    ctx.textBaseline = 'middle';

    for (let midi = Math.ceil(scale.lo); midi <= Math.floor(scale.hi); midi++) {
        const y = scale.freqToY(midiToFreq(midi), h);
        const note = ((midi % 12) + 12) % 12;
        const octave = Math.floor(midi / 12) - 1;
        const inScale = hasScale && (noteBits & (1 << note)) !== 0;
        const isRoot = hasScale && note === root;
        const isC = note === 0;

        // Line emphasis.
        let alpha;
        if (hasScale) {
            if (isRoot) alpha = 0.34;
            else if (isC) alpha = 0.22;
            else if (inScale) alpha = 0.16;
            else alpha = semitonePx < 5 * dpr ? 0 : 0.05;
        } else {
            if (isC) alpha = 0.28;
            else if (note === 7) alpha = 0.14;
            else alpha = semitonePx < 5 * dpr ? 0 : 0.07;
        }
        if (alpha > 0) {
            ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Label density tiers (10px text needs ~13px rows to not collide).
        let label = null;
        if (semitonePx >= 13 * dpr) {
            if (!hasScale || inScale || isC) label = `${NOTE_NAMES[note]}${octave}`;
        } else if (semitonePx >= 6 * dpr) {
            if (isRoot || isC) label = `${NOTE_NAMES[note]}${octave}`;
        } else if (isC) {
            label = `C${octave}`;
        }
        if (label) {
            ctx.fillStyle = `rgba(255,255,255,${isRoot || isC ? 0.75 : 0.5})`;
            // Centered on the line; clamped so edge labels stay readable.
            const ty = Math.max(6 * dpr, Math.min(h - 6 * dpr, y));
            ctx.fillText(label, 4 * dpr, ty);
        }
    }
}

/**
 * Draw a pitch track (Hz per hop, 0 = unvoiced) for the whole viewport.
 * vp: {s0, spp, w, h, dpr} in samples / device px.
 */
export function drawPitchTrack(ctx, track, hopSamples, vp, color, scale) {
    const { s0, spp, w, h, dpr } = vp;
    const spacingCss = hopSamples / spp / dpr;
    const first = Math.max(0, Math.floor(s0 / hopSamples));
    const last = Math.min(track.length - 1, Math.ceil((s0 + w * spp) / hopSamples));

    if (spacingCss < DENSE_LIMIT_CSS) {
        // Aggregate voiced hops into per-column min/max.
        const cols = [];
        let cur = null;
        for (let i = first; i <= last; i++) {
            const y = scale.freqToY(track[i], h);
            if (y < 0 || y > h) continue;
            const x = Math.floor((i * hopSamples - s0) / spp);
            if (cur && cur.x === x) {
                cur.y0 = Math.min(cur.y0, y);
                cur.y1 = Math.max(cur.y1, y);
            } else {
                cur = { x, y0: y, y1: y };
                cols.push(cur);
            }
        }
        drawEnvelopeColumns(ctx, cols, dpr, color);
    } else {
        const pts = [];
        for (let i = first; i <= last; i++) {
            const y = scale.freqToY(track[i], h);
            pts.push(y < 0 || y > h ? null : { x: (i * hopSamples - s0) / spp, y });
        }
        drawLinesAndDots(ctx, pts, spacingCss, dpr, color);
    }
}

// --- Waveform view: linear amplitude, one point per sample ---

/**
 * Draw a waveform for the viewport. fetchPeaks(start, end, bins) returns
 * bins*2 floats interleaved [min, max] (from Rust). Dense zoom renders the
 * min/max envelope incrementally over device-px columns [x0, x1); sparse
 * (sample-level) zoom fetches one bin per sample and plots points, always
 * repainting the full width since lines span columns.
 * vp: {s0, spp, w, h, dpr, end} — end is the absolute data end in samples.
 */
export function drawWaveform(ctx, fetchPeaks, vp, color, x0, x1) {
    const { s0, spp, w, h, dpr } = vp;
    const mid = h / 2;
    const spacingCss = 1 / spp / dpr;

    if (spacingCss < DENSE_LIMIT_CSS) {
        ctx.fillStyle = BG;
        ctx.fillRect(x0, 0, x1 - x0, h);
        const bins = x1 - x0;
        if (bins <= 0) return;
        const peaks = fetchPeaks(s0 + x0 * spp, s0 + x1 * spp, bins);
        const cols = [];
        for (let i = 0; i < bins; i++) {
            const mn = peaks[i * 2];
            const mx = peaks[i * 2 + 1];
            if (mn === 0 && mx === 0) continue;
            cols.push({ x: x0 + i, y0: mid - mx * mid, y1: mid - mn * mid });
        }
        drawEnvelopeColumns(ctx, cols, dpr, color);
    } else {
        ctx.fillStyle = BG;
        ctx.fillRect(0, 0, w, h);
        const a = Math.max(0, Math.floor(s0));
        const b = Math.min(Math.ceil(s0 + w * spp), vp.end ?? Infinity);
        const n = b - a;
        if (n <= 0) return;
        const peaks = fetchPeaks(a, b, n); // one bin per sample: min == max
        const pts = [];
        for (let j = 0; j < n; j++) {
            pts.push({ x: (a + j - s0) / spp, y: mid - peaks[j * 2] * mid });
        }
        drawLinesAndDots(ctx, pts, spacingCss, dpr, color);
    }
}
