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

// --- Pitch view: log-frequency axis, C2..C6 ---

export const PITCH_FMIN = 65.406;   // C2
export const PITCH_FMAX = 1046.502; // C6

const LOG_FMIN = Math.log2(PITCH_FMIN);
const LOG_FMAX = Math.log2(PITCH_FMAX);

/** Frequency (Hz) -> y device px. Top = high pitch. NaN-safe for f<=0. */
export function freqToY(freq, height) {
    if (freq <= 0) return -1;
    const t = (Math.log2(freq) - LOG_FMIN) / (LOG_FMAX - LOG_FMIN);
    return (1 - t) * height;
}

/** y device px -> frequency (Hz). */
export function yToFreq(y, height) {
    const t = 1 - y / height;
    return 2 ** (LOG_FMIN + t * (LOG_FMAX - LOG_FMIN));
}

/** Background + semitone/octave grid + note labels for a pitch track. */
export function drawPitchGrid(ctx, w, h, dpr) {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    // Semitones C2 (midi 36) .. C6 (midi 84).
    const semitonePx = h / (4 * 12);
    for (let midi = 36; midi <= 84; midi++) {
        const freq = 440 * 2 ** ((midi - 69) / 12);
        const y = freqToY(freq, h);
        const note = midi % 12;
        if (note === 0) {
            ctx.strokeStyle = 'rgba(255,255,255,0.28)';
        } else if (semitonePx < 5 * dpr && note !== 7) {
            continue; // too dense: only octaves and fifths
        } else {
            ctx.strokeStyle = note === 7 ? 'rgba(255,255,255,0.14)' : 'rgba(255,255,255,0.07)';
        }
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
        if (note === 0) {
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.font = `${10 * dpr}px monospace`;
            ctx.fillText(`C${midi / 12 - 1}`, 4 * dpr, y - 3 * dpr);
        }
    }
}

/**
 * Draw a pitch track (Hz per hop, 0 = unvoiced) for the whole viewport.
 * vp: {s0, spp, w, h, dpr} in samples / device px.
 */
export function drawPitchTrack(ctx, track, hopSamples, vp, color) {
    const { s0, spp, w, h, dpr } = vp;
    const spacingCss = hopSamples / spp / dpr;
    const first = Math.max(0, Math.floor(s0 / hopSamples));
    const last = Math.min(track.length - 1, Math.ceil((s0 + w * spp) / hopSamples));

    if (spacingCss < DENSE_LIMIT_CSS) {
        // Aggregate voiced hops into per-column min/max.
        const cols = [];
        let cur = null;
        for (let i = first; i <= last; i++) {
            const y = freqToY(track[i], h);
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
            const y = freqToY(track[i], h);
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
