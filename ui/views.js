// Track content renderers for pitch-contour and waveform views, plus the
// log-frequency axis used by the pitch view. (Spectrogram pixels come from
// Rust via draw_*_spectrogram_range; these are the light JS-side views.)

// Log-frequency pitch axis: C2..C6 covers the vocal range with headroom.
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

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/** Background + semitone/octave grid + note labels for a pitch track. */
export function drawPitchGrid(ctx, w, h, dpr) {
    ctx.fillStyle = 'rgb(10,10,20)';
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
 * Draw a pitch track (Hz per hop, 0 = unvoiced) for the viewport.
 * vp: {s0, spp, w, h} in samples / device px.
 */
export function drawPitchTrack(ctx, track, hopSamples, vp, color, dpr) {
    const { s0, spp, w, h } = vp;
    ctx.fillStyle = color;
    const r = 1.5 * dpr;
    const hopPx = hopSamples / spp;
    // Iterate hops visible in the viewport.
    const first = Math.max(0, Math.floor(s0 / hopSamples));
    const last = Math.min(track.length - 1, Math.ceil((s0 + w * spp) / hopSamples));
    ctx.beginPath();
    for (let i = first; i <= last; i++) {
        const freq = track[i];
        if (freq <= 0) continue;
        const y = freqToY(freq, h);
        if (y < 0 || y > h) continue;
        const x = (i * hopSamples - s0) / spp;
        if (hopPx > 2 * r) {
            ctx.moveTo(x + r, y);
            ctx.arc(x, y, r, 0, Math.PI * 2);
        } else {
            // Zoomed out: cheap 2px marks instead of thousands of arcs.
            ctx.rect(x, y - dpr, Math.max(hopPx, dpr), 2 * dpr);
        }
    }
    ctx.fill();
}

/** Min/max peaks waveform (peaks: [min,max] per column starting at x0). */
export function drawWaveform(ctx, peaks, x0, w, h, color) {
    ctx.fillStyle = 'rgb(10,10,20)';
    ctx.fillRect(x0, 0, w, h);
    const mid = h / 2;
    ctx.fillStyle = color;
    for (let i = 0; i < peaks.length / 2; i++) {
        const mn = peaks[i * 2];
        const mx = peaks[i * 2 + 1];
        if (mn === 0 && mx === 0) continue;
        const yTop = mid - mx * mid;
        const yBot = mid - mn * mid;
        ctx.fillRect(x0 + i, yTop, 1, Math.max(1, yBot - yTop));
    }
}
