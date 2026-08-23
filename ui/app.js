import init, { WebPitchCorrector, warmup } from '../pkg/pitch_corrector.js';
import { Timeline } from './timeline.js';
import {
    AmplitudeScale,
    drawPitchGrid,
    drawPitchTrack,
    drawWaveform,
    PitchScale,
} from './views.js';
import { encodeWav, decodeWav, downloadBlob } from './wav.js';

// --- State ---
// Transport: idle -> recording -> stopped <-> playing <-> paused
let state = 'idle';
let corrector = null;
let sampleRate = 48000;
let totalSamples = 0;
let pitchHop = 1024;
let vocoderHop = 512;

// Post-correction: target contour (one entry per phase-vocoder hop, spanning
// the whole recording) captured at stop, plus the user-edited copy.
let targetContour = [];
let editedContour = null;
let postCorrectionActive = false;

// One shared y axis for every pitch view (input, output, contour editor),
// dynamically fitted to the data with hysteresis. See PitchScale.
const pitchScale = new PitchScale();

// Vertical-axis zoom state (ctrl+scroll), shared between the two tracks so
// they stay comparable: the spectrogram's log-frequency window (fractions
// of the full axis). Waveform gain lives on ampScale, pitch on pitchScale.
const vzoom = {
    spec: { lo: 0, hi: 1 },
};

// Auto-fitting waveform gain, shared by both tracks (like the pitch axis).
const ampScale = new AmplitudeScale();

const INPUT_COLOR = 'rgb(255,150,50)';
const OUTPUT_COLOR = 'rgb(50,255,120)';
// Merged pitch view: which series are drawn (legend toggles).
const pitchVis = { input: true, output: true, aim: true, h1: true, h2: true, h3: true };
// The aim line (post-smoothing, full strength): output green, dimmed.
const AIM_COLOR = 'rgba(50,255,120,0.35)';
const EDIT_COLOR = 'rgb(255,80,200)';
// Harmony voices get their own contours on the output pitch view (their
// pitch comes from the DSP's per-voice logs, never from running a
// monophonic detector on the mixed output): 3rd, 5th, octave.
const HARMONY_COLORS = ['rgba(90,190,255,0.9)', 'rgba(200,130,255,0.9)', 'rgba(255,220,90,0.9)'];

const $ = (id) => document.getElementById(id);
/** @returns {HTMLButtonElement} */
const $btn = (id) => /** @type {HTMLButtonElement} */ ($(id));
/** @returns {HTMLInputElement} */
const $input = (id) => /** @type {HTMLInputElement} */ ($(id));
/** @returns {HTMLSelectElement} */
const $select = (id) => /** @type {HTMLSelectElement} */ ($(id));
const els = {
    status: $('status'),
    recordBtn: $btn('record-btn'),
    stopBtn: $btn('stop-btn'),
    playBtn: $btn('play-btn'),
    passthroughBtn: $btn('passthrough-btn'),
    dryBtn: $btn('dry-btn'),
    downloadBtn: $btn('download-btn'),
    debugBtn: $btn('debug-btn'),
    uploadBtn: $btn('upload-btn'),
    uploadInput: $input('upload-input'),
    postCorrectionLabel: $('post-correction-label'),
    postCorrectionCb: $input('post-correction-cb'),
};

// --- Timeline ---

const timeline = new Timeline($('timeline'), {
    tracks: [
        {
            id: 'input',
            label: 'Input',
            views: ['waveform', 'pitch', 'spectrogram'],
            view: 'waveform',
        },
        {
            id: 'output',
            label: 'Output',
            views: ['waveform', 'pitch', 'spectrogram'],
            view: 'waveform',
        },
    ],
    renderTrack,
    // Spectrogram columns cost an FFT each; budget them per frame so big
    // repaints stay off the long-task radar. Other views are cheap.
    renderBudget: (t) => (t.view === 'spectrogram' ? 128 : Infinity),
    onSeek: seekTo,
    onViewChange: () => updatePostCorrectionVisibility(),
    onTrackDrag: contourDrag,
    onVerticalZoom: verticalZoom,
    onFit: resetVerticalZoom,
});
// E2E test hook: expose the timeline for viewport assertions.
if (new URLSearchParams(location.search).has('e2e')) {
    const w = /** @type {any} */ (window);
    w.__tl = timeline;
    w.__scale = pitchScale;
    w.__amp = ampScale;
}

// --- View selection: one shared dropdown by default, split on demand ---

const viewSelect = $select('view-select');
const splitCb = $input('split-views-cb');

// The shared pitch view merges every series into one tall lane with a
// legend; split-per-track does not apply to it (each lane would only
// duplicate the merged content), so the checkbox hides while it is active.
function pitchMerged() {
    return !splitCb.checked && viewSelect.value === 'pitch';
}
function updatePitchMerge() {
    timeline.collapseTo(pitchMerged() ? 'input' : null);
    legendEl.style.display = pitchMerged() ? '' : 'none';
    splitLabel.style.display = viewSelect.value === 'pitch' ? 'none' : '';
}
function applySharedView() {
    timeline.setView('input', viewSelect.value);
    timeline.setView('output', viewSelect.value);
    updatePitchMerge();
}
viewSelect.addEventListener('change', applySharedView);
splitCb.addEventListener('change', () => {
    timeline.showTrackSelectors(splitCb.checked);
    viewSelect.disabled = splitCb.checked;
    if (!splitCb.checked) applySharedView(); // re-unify on un-split
    updatePitchMerge();
});

// Legend: top-right overlay on the merged pitch lane, one switch per series.
const splitLabel = /** @type {HTMLElement} */ (splitCb.parentElement);
const legendEl = document.createElement('div');
legendEl.className = 'pitch-legend';
legendEl.style.display = 'none';
for (const [key, label, color] of [
    ['input', 'Input', INPUT_COLOR],
    ['output', 'Output', OUTPUT_COLOR],
    ['aim', 'Aim', AIM_COLOR],
    ['h1', '3rd', HARMONY_COLORS[0]],
    ['h2', '5th', HARMONY_COLORS[1]],
    ['h3', 'Octave', HARMONY_COLORS[2]],
]) {
    const row = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = pitchVis[key];
    cb.dataset.series = key;
    cb.addEventListener('change', () => {
        pitchVis[key] = cb.checked;
        timeline.invalidate('input');
    });
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = color;
    row.append(cb, swatch, document.createTextNode(label));
    legendEl.appendChild(row);
}
timeline.getTrack('input').canvas.parentElement.appendChild(legendEl);

function renderTrack(track, canvas, vp, x0, x1) {
    const isInput = track.id === 'input';
    const ctx = canvas.getContext('2d');
    if (!corrector || totalSamples === 0) {
        ctx.fillStyle = 'rgb(10,10,20)';
        ctx.fillRect(0, 0, vp.w, vp.h);
        return;
    }
    switch (track.view) {
        case 'spectrogram': {
            const w = x1 - x0;
            const start = vp.s0 + x0 * vp.spp;
            if (isInput) {
                corrector.draw_input_spectrogram_range(
                    canvas,
                    x0,
                    w,
                    start,
                    vp.spp,
                    vzoom.spec.lo,
                    vzoom.spec.hi,
                );
            } else {
                corrector.draw_output_spectrogram_range(
                    canvas,
                    x0,
                    w,
                    start,
                    vp.spp,
                    vzoom.spec.lo,
                    vzoom.spec.hi,
                );
            }
            break;
        }
        case 'pitch': {
            // Cheap enough to always repaint fully.
            drawPitchGrid(ctx, vp.w, vp.h, vp.dpr, pitchScale, {
                noteBits: corrector.get_scale(),
                root: parseInt($select('root-select').value),
            });
            if (pitchMerged()) {
                // One lane, every series overlaid; the legend gates each.
                for (let v = 1; v <= 3; v++) {
                    if (!pitchVis['h' + v]) continue;
                    const track = corrector.harmony_pitch_track(v);
                    if (track.some((f) => f > 0)) {
                        drawPitchTrack(
                            ctx,
                            track,
                            vocoderHop,
                            vp,
                            HARMONY_COLORS[v - 1],
                            pitchScale,
                        );
                    }
                }
                if (pitchVis.aim) {
                    drawPitchTrack(
                        ctx,
                        corrector.aim_pitch_track(),
                        vocoderHop,
                        vp,
                        AIM_COLOR,
                        pitchScale,
                    );
                }
                if (pitchVis.input) {
                    drawPitchTrack(
                        ctx,
                        corrector.input_pitch_track(),
                        pitchHop,
                        vp,
                        INPUT_COLOR,
                        pitchScale,
                    );
                }
                if (pitchVis.output) {
                    drawPitchTrack(
                        ctx,
                        corrector.output_pitch_track(),
                        vocoderHop,
                        vp,
                        OUTPUT_COLOR,
                        pitchScale,
                    );
                }
                if (postCorrectionActive && editedContour) {
                    drawEditedContour(ctx, vp);
                }
            } else if (isInput) {
                drawPitchTrack(
                    ctx,
                    corrector.input_pitch_track(),
                    pitchHop,
                    vp,
                    INPUT_COLOR,
                    pitchScale,
                );
            } else {
                // Harmony voices first (dimmer, underneath), main on top.
                for (let v = 1; v <= 3; v++) {
                    const track = corrector.harmony_pitch_track(v);
                    if (track.some((f) => f > 0)) {
                        drawPitchTrack(
                            ctx,
                            track,
                            vocoderHop,
                            vp,
                            HARMONY_COLORS[v - 1],
                            pitchScale,
                        );
                    }
                }
                // The corrector's aim (post-smoothing, full strength):
                // only distinct from the output when strength < 1.
                if (parseFloat(strengthSlider.value) < 100) {
                    drawPitchTrack(
                        ctx,
                        corrector.aim_pitch_track(),
                        vocoderHop,
                        vp,
                        AIM_COLOR,
                        pitchScale,
                    );
                }
                drawPitchTrack(
                    ctx,
                    corrector.output_pitch_track(),
                    vocoderHop,
                    vp,
                    OUTPUT_COLOR,
                    pitchScale,
                );
                if (postCorrectionActive && editedContour) {
                    drawEditedContour(ctx, vp);
                }
            }
            break;
        }
        case 'waveform': {
            const fetchPeaks = isInput
                ? (a, b, n) => corrector.input_peaks(a, b, n)
                : (a, b, n) => corrector.output_peaks(a, b, n);
            drawWaveform(
                ctx,
                fetchPeaks,
                vp,
                isInput ? INPUT_COLOR : OUTPUT_COLOR,
                x0,
                x1,
                ampScale.gain,
            );
            break;
        }
    }
}

function drawEditedContour(ctx, vp) {
    // Contour entry i belongs at input sample i * vocoderHop — hop-true,
    // exactly like the voice pitch logs. Never rescale by the recording
    // length: the log is bounded, so it may be shorter than the recording,
    // and rescaling would smear every entry off its true position.
    const n = editedContour.length;
    if (n === 0) return;
    ctx.fillStyle = EDIT_COLOR;
    const first = Math.max(0, Math.floor(vp.s0 / vocoderHop));
    const last = Math.min(n - 1, Math.ceil((vp.s0 + vp.w * vp.spp) / vocoderHop));
    for (let i = first; i <= last; i++) {
        const freq = editedContour[i];
        if (freq <= 0) continue;
        const y = pitchScale.freqToY(freq, vp.h);
        if (y < 0 || y > vp.h) continue;
        const x = (i * vocoderHop - vp.s0) / vp.spp;
        ctx.fillRect(x - vp.dpr, y - vp.dpr, 2 * vp.dpr, 2 * vp.dpr);
    }
}

// Fit shows everything: restore the vertical axes alongside the timeline.
function resetVerticalZoom() {
    vzoom.spec = { lo: 0, hi: 1 };
    pitchScale.reset();
    ampScale.reset();
    updateViewScales(); // auto-fit is back: refit to the data now
    timeline.invalidate();
}

// Ctrl+scroll: zoom the vertical axis of whichever view is under the
// cursor. Shared between tracks so input and output stay comparable.
function verticalZoom(track, factor, yFrac) {
    switch (track.view) {
        case 'pitch':
            pitchScale.zoomBy(factor, yFrac);
            break;
        case 'waveform':
            ampScale.zoomBy(factor); // pins manual; Fit resumes auto
            break;
        case 'spectrogram': {
            const span = Math.min(Math.max((vzoom.spec.hi - vzoom.spec.lo) * factor, 0.05), 1);
            const anchor = vzoom.spec.hi - yFrac * (vzoom.spec.hi - vzoom.spec.lo);
            let hi = anchor + yFrac * span;
            let lo = hi - span;
            if (lo < 0) {
                hi -= lo;
                lo = 0;
            }
            if (hi > 1) {
                lo -= hi - 1;
                hi = 1;
            }
            vzoom.spec = { lo, hi };
            break;
        }
    }
    for (const id of ['input', 'output']) {
        if (timeline.getTrack(id).view === track.view) timeline.invalidate(id);
    }
}

// Drag on the output pitch view edits the target contour.
function contourDrag(track, pos, phase) {
    if (track.view !== 'pitch') return false;
    // Editing lives on the output lane, or on the merged lane (which is
    // hosted by the input track) when the pitch view is unified.
    if (track.id !== 'output' && !pitchMerged()) return false;
    if (!postCorrectionActive || !editedContour || totalSamples === 0) return false;
    if (state !== 'stopped' && state !== 'paused') return false;
    if (phase === 'end') return true;

    const hop = Math.floor(pos.sample / vocoderHop);
    if (hop < 0 || hop >= editedContour.length) return true;
    const freq = pitchScale.yToFreq(pos.y, pos.h);
    const noteBits = corrector.get_scale();
    editedContour[hop] = noteBits > 0 ? WebPitchCorrector.snap_to_scale(freq, noteBits) : freq;
    timeline.invalidate(pitchMerged() ? 'input' : 'output');
    return true;
}

function invalidatePitchViews() {
    for (const id of ['input', 'output']) {
        if (timeline.getTrack(id).view === 'pitch') timeline.invalidate(id);
    }
}

// Refit the shared view scales (pitch axis, waveform gain) to the current
// data; on a real change (both have hysteresis) repaint the affected views.
function updateViewScales() {
    if (!corrector) return;
    const pitchChanged = pitchScale.update([
        corrector.input_pitch_track(),
        corrector.output_pitch_track(),
        corrector.harmony_pitch_track(1),
        corrector.harmony_pitch_track(2),
        corrector.harmony_pitch_track(3),
    ]);
    if (pitchChanged) invalidatePitchViews();

    if (totalSamples > 0) {
        const bins = 512;
        const ampChanged = ampScale.update([
            corrector.input_peaks(0, totalSamples, bins),
            corrector.output_peaks(0, totalSamples, bins),
        ]);
        if (ampChanged) {
            for (const id of ['input', 'output']) {
                if (timeline.getTrack(id).view === 'waveform') timeline.invalidate(id);
            }
        }
    }
}

// --- Render loop ---
// One always-on rAF loop; Timeline.render() early-outs when nothing changed.
let scaleThrottle = 0;
function loop() {
    requestAnimationFrame(loop);
    if (!corrector) return;
    if (state === 'recording' || state === 'playing') {
        if (++scaleThrottle % 15 === 0) updateViewScales(); // ~4x/s is plenty
        totalSamples = corrector.analyze();
        timeline.setTotal(totalSamples);
        // The output lags the input by the pipeline latency (and regrows
        // from the seek position during playback): repaint from its end.
        timeline.setDataEnd('output', corrector.output_len());
        if (state === 'recording') {
            timeline.setPlayhead(totalSamples);
        } else {
            timeline.setPlayhead(corrector.playback_progress() * totalSamples);
            if (!corrector.is_playing()) {
                // Playback reached the end: pause the output stream too —
                // left running it pulls from an unfed pipeline forever
                // (an endless stream of underruns).
                corrector.stop_playback();
                corrector.clear_contour();
                setState('stopped');
            }
        }
        timeline.render();
    }
}
requestAnimationFrame(loop);

// --- Transport state machine ---

function setState(newState) {
    if (corrector && corrector.is_freezing() && newState !== 'stopped' && newState !== 'paused') {
        corrector.stop_freeze();
    }
    state = newState;
    const s = {
        idle: 'Click Record to begin',
        recording: '🔴 Recording…',
        stopped: 'Recording saved. Play it back, zoom with the mouse wheel, click to seek.',
        playing: '▶ Playing…',
        paused: 'Paused. Play to resume, click a track to seek.',
    };
    els.status.textContent = s[state];
    els.recordBtn.disabled = state === 'recording' || state === 'playing';
    els.stopBtn.disabled = state !== 'recording';
    els.playBtn.disabled =
        !(state === 'stopped' || state === 'paused' || state === 'playing') || totalSamples === 0;
    els.playBtn.textContent = state === 'playing' ? '⏸ Pause' : '▶ Play';
    const hasRecording = totalSamples > 0 && (state === 'stopped' || state === 'paused');
    els.downloadBtn.disabled = !hasRecording;
    els.debugBtn.disabled = !hasRecording;
    updatePostCorrectionVisibility();
}

function checkBrowserSupport() {
    const missing = [];
    if (typeof WebAssembly === 'undefined') missing.push('WebAssembly');
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia))
        missing.push('microphone access (getUserMedia)');
    if (!(window.AudioContext || /** @type {any} */ (window).webkitAudioContext))
        missing.push('Web Audio');
    return missing;
}

function describeStartError(e) {
    const s = ((e && (e.name || '')) + ' ' + (e && (e.message || ''))).trim() || String(e);
    const hay = s.toLowerCase();
    if (hay.includes('notallowed') || hay.includes('permission') || hay.includes('denied'))
        return 'Microphone permission denied. Allow mic access in your browser, then click Record again.';
    if (
        hay.includes('notfound') ||
        hay.includes('no input device') ||
        hay.includes('devicesnotfound')
    )
        return 'No microphone found. Connect a microphone and click Record again.';
    if (hay.includes('notreadable') || hay.includes('in use'))
        return 'Microphone is in use by another application. Close it and retry.';
    return 'Could not start audio: ' + s;
}

// Resolve once the whole audio pipeline (both worklets) is live, or after a
// timeout (e.g. mic denied) so we never hang.
function waitForAudioReady(c, timeoutMs = 6000) {
    return new Promise((resolve) => {
        const t0 = performance.now();
        (function poll() {
            if (c.is_audio_ready() || performance.now() - t0 > timeoutMs) resolve(undefined);
            else setTimeout(poll, 30);
        })();
    });
}

function newCorrector() {
    // One corrector (one audio graph) for the whole session: cpal's web
    // hosts don't reliably stop a replaced stream's callbacks, so
    // constructing a second WebPitchCorrector leaves the old AudioContext
    // live and fighting over the microphone. Re-record via start_recording.
    if (corrector) return;
    corrector = new WebPitchCorrector();
    // E2E test hook: expose the corrector so Playwright can inspect the
    // captured buffers. Only when ?e2e is set.
    if (new URLSearchParams(location.search).has('e2e'))
        /** @type {any} */ (window).__pc = corrector;
    sampleRate = corrector.sample_rate();
    pitchHop = corrector.pitch_hop();
    vocoderHop = corrector.vocoder_hop();
    timeline.setSampleRate(sampleRate);
}

function resetSession() {
    pitchScale.reset(); // manual vertical zoom ends with the session
    totalSamples = 0;
    targetContour = [];
    editedContour = null;
    postCorrectionActive = false;
    els.postCorrectionCb.checked = false;
    timeline.reset();
}

async function startRecording() {
    const missing = checkBrowserSupport();
    if (missing.length) {
        els.status.textContent =
            'Unsupported browser — missing ' +
            missing.join(', ') +
            '. Try a recent Chrome, Edge, or Firefox over HTTPS.';
        return;
    }
    try {
        await init();
    } catch (e) {
        setState('idle');
        els.status.textContent =
            'Failed to load the audio engine (WASM). Check your connection and reload. (' + e + ')';
        return;
    }
    try {
        if (!corrector) {
            // Ask for the microphone explicitly, first thing: the
            // getUserMedia promise resolves only once the permission
            // prompt is accepted (and rejects on deny), so we don't build
            // the audio graph or start drawing while the prompt is open.
            // The probe track is released immediately; cpal re-acquires
            // with the now-granted permission.
            els.status.textContent = 'Waiting for microphone permission…';
            const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
            probe.getTracks().forEach((t) => t.stop());
        }
        newCorrector();
        resetSession();
        applyScale();
        corrector.set_shift(parseFloat(slider.value));
        corrector.set_monitor(passthroughOn);
        corrector.set_bypass(dryOn);
        applyRetune();
        applyStrength();
        applyHarmony();
        corrector.start_recording();

        els.status.textContent = 'Initializing audio…';
        await waitForAudioReady(corrector);
        if (!corrector.is_audio_ready()) {
            throw new Error('The audio pipeline did not start in time. Reload and try again.');
        }

        timeline.follow(true);
        setState('recording');
    } catch (e) {
        // Order matters: setState writes the idle status line; the error
        // message must land after it.
        setState('idle');
        els.status.textContent = describeStartError(e);
    }
}

function stopRecording() {
    if (state !== 'recording') return;
    corrector.stop();
    totalSamples = corrector.analyze();
    timeline.setTotal(totalSamples);
    if (totalSamples === 0) {
        setState('idle');
        return;
    }
    targetContour = Array.from(corrector.take_target_pitch_contour());
    updateViewScales();
    timeline.follow(false);
    timeline.fit();
    timeline.setPlayhead(0);
    corrector.seek(0);
    setState('stopped');
}

function startPlayback() {
    if (!corrector || totalSamples === 0) return;
    if (corrector.playback_progress() >= 1.0) corrector.seek(0);
    if (postCorrectionActive && editedContour) {
        corrector.set_contour(new Float32Array(editedContour));
    } else {
        corrector.clear_contour();
    }
    corrector.play_recording();
    // Rust truncated the output at the play position; roll the repaint
    // watermark back so the re-processed audio gets drawn.
    timeline.setDataEnd('output', corrector.output_len());
    setState('playing');
}

function pausePlayback() {
    corrector.stop_playback();
    corrector.clear_contour();
    setState('paused');
}

function seekTo(sample) {
    if (!corrector || totalSamples === 0) return;
    if (state !== 'stopped' && state !== 'playing' && state !== 'paused') return;
    corrector.seek(sample / totalSamples);
    timeline.setPlayhead(sample);
    timeline.setDataEnd('output', corrector.output_len());
    timeline.render();
}

els.recordBtn.addEventListener('click', startRecording);

// Dry bypass (A/B): hear the uncorrected voice. Toggles live — the DSP
// glides the correction to unity and fades harmonies out, during both
// recording and playback re-processing.
let dryOn = false;
els.dryBtn.addEventListener('click', () => {
    dryOn = !dryOn;
    els.dryBtn.classList.toggle('active', dryOn);
    if (corrector) corrector.set_bypass(dryOn);
});

// Live passthrough: hear the corrected output while recording. Off by
// default — the DSP and visuals run either way, only the speakers gate.
let passthroughOn = false;
els.passthroughBtn.addEventListener('click', () => {
    passthroughOn = !passthroughOn;
    els.passthroughBtn.classList.toggle('active', passthroughOn);
    if (corrector) corrector.set_monitor(passthroughOn);
});
els.stopBtn.addEventListener('click', stopRecording);
els.playBtn.addEventListener('click', () => {
    if (state === 'playing') pausePlayback();
    else if (state === 'stopped' || state === 'paused') startPlayback();
});
window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        // Don't fight a focused control (Space also activates buttons).
        const tag = /** @type {HTMLElement} */ (e.target).tagName;
        if (tag === 'BUTTON' || tag === 'SELECT') return;
        if (state === 'playing') {
            e.preventDefault();
            pausePlayback();
        } else if (state === 'stopped' || state === 'paused') {
            e.preventDefault();
            startPlayback();
        }
    } else if (e.code === 'KeyH' && !e.repeat) {
        // Hold h: spectral-freeze audition of the frame under the playhead.
        // During playback, pause first and freeze right there.
        if (!corrector || totalSamples === 0) return;
        if (state === 'playing') pausePlayback();
        if (state !== 'stopped' && state !== 'paused') return;
        e.preventDefault();
        const pos = corrector.playback_progress() * totalSamples;
        if (corrector.start_freeze(pos)) {
            els.status.textContent =
                '❄ Sustaining the frame under the playhead (release h to stop)';
        }
    }
});
window.addEventListener('keyup', (e) => {
    if (e.code === 'KeyH' && corrector && corrector.is_freezing()) {
        corrector.stop_freeze();
        setState(state); // restore the state's status line
    }
});

// --- Harmony controls ---
// Voices are a bitmask (bit0=3rd, bit1=5th, bit2=octave); intervals are
// either diatonic (walk the selected scale from the corrected note) or
// absolute (fixed semitone offsets). Defaults: off, in key.
let harmonyMask = 0;
let harmonyInKey = true;

function applyHarmony() {
    if (!corrector) return;
    corrector.set_harmony(harmonyMask);
    corrector.set_harmony_in_key(harmonyInKey);
}

const harmonyOffBtn = $btn('harmony-off');
harmonyOffBtn.addEventListener('click', () => {
    harmonyMask = 0;
    document
        .querySelectorAll('#harmony-controls button[data-hvoice]')
        .forEach((b) => b.classList.remove('active'));
    harmonyOffBtn.classList.add('active');
    applyHarmony();
});
/** @type {NodeListOf<HTMLButtonElement>} */ (
    document.querySelectorAll('#harmony-controls button[data-hvoice]')
).forEach((btn) => {
    btn.addEventListener('click', () => {
        harmonyMask ^= 1 << parseInt(btn.dataset.hvoice);
        btn.classList.toggle('active');
        harmonyOffBtn.classList.toggle('active', harmonyMask === 0);
        applyHarmony();
    });
});
/** @type {NodeListOf<HTMLButtonElement>} */ (
    document.querySelectorAll('#harmony-controls button[data-hmode]')
).forEach((btn) => {
    btn.addEventListener('click', () => {
        document
            .querySelectorAll('#harmony-controls button[data-hmode]')
            .forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        harmonyInKey = btn.dataset.hmode === 'key';
        applyHarmony();
    });
});

// --- Scale / shift controls ---

const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
let currentScale = 'pentatonic';
let customBits = 0xfff;

const noteGrid = $('note-grid');
noteNames.forEach((name, i) => {
    const btn = document.createElement('button');
    btn.textContent = name;
    btn.dataset.note = String(i);
    btn.classList.add('active');
    btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        customBits = 0;
        /** @type {NodeListOf<HTMLButtonElement>} */ (
            noteGrid.querySelectorAll('button.active')
        ).forEach((b) => {
            customBits |= 1 << parseInt(b.dataset.note);
        });
        if (corrector && currentScale === 'custom') {
            corrector.set_scale(customBits);
            invalidatePitchViews();
        }
    });
    noteGrid.appendChild(btn);
});

function applyScale() {
    if (!corrector) return;
    const root = parseInt($select('root-select').value);
    if (currentScale === 'custom') {
        corrector.set_scale(customBits);
    } else {
        corrector.set_scale(WebPitchCorrector.scale_bits(currentScale, root));
    }
    invalidatePitchViews(); // grid emphasis and labels follow the scale
}

/** @type {NodeListOf<HTMLButtonElement>} */ (
    document.querySelectorAll('#scale-buttons button')
).forEach((btn) => {
    btn.addEventListener('click', () => {
        document
            .querySelectorAll('#scale-buttons button')
            .forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        currentScale = btn.dataset.scale;
        $('custom-notes').style.display = currentScale === 'custom' ? 'block' : 'none';
        applyScale();
    });
});

$select('root-select').addEventListener('change', applyScale);

const slider = $input('shift-slider');
const shiftDisplay = $('shift-display');
function updateShiftDisplay() {
    const v = parseFloat(slider.value);
    shiftDisplay.textContent = (v >= 0 ? '+' : '') + v + ' st';
}
slider.addEventListener('input', () => {
    updateShiftDisplay();
    if (corrector) corrector.set_shift(parseFloat(slider.value));
});

// Retune speed: log-scale slider position 0..100 -> 1..500 ms.
const retuneSlider = $input('retune-slider');
const retuneDisplay = $('retune-display');
function retuneMs() {
    return 10 ** ((parseFloat(retuneSlider.value) / 100) * Math.log10(500));
}
function applyRetune() {
    retuneDisplay.textContent = Math.round(retuneMs()) + ' ms';
    if (corrector) corrector.set_retune_speed_ms(retuneMs());
}
retuneSlider.addEventListener('input', applyRetune);

const strengthSlider = $input('strength-slider');
const strengthDisplay = $('strength-display');
function applyStrength() {
    strengthDisplay.textContent = strengthSlider.value + '%';
    if (corrector) corrector.set_strength(parseFloat(strengthSlider.value) / 100);
}
strengthSlider.addEventListener('input', applyStrength);

// --- Post-correction ---

function updatePostCorrectionVisibility() {
    const show = (state === 'stopped' || state === 'paused') && targetContour.length > 0;
    els.postCorrectionLabel.style.display = show ? '' : 'none';
}

els.postCorrectionCb.addEventListener('change', () => {
    postCorrectionActive = els.postCorrectionCb.checked;
    if (postCorrectionActive && !editedContour) editedContour = targetContour.slice();
    if (postCorrectionActive) {
        // Editing happens on the output pitch view — switch to it.
        if (splitCb.checked) {
            timeline.setView('output', 'pitch');
        } else {
            viewSelect.value = 'pitch';
            applySharedView();
        }
        els.status.textContent =
            'Post-correction: drag on the pitch view to edit the melody, then Play.';
    }
    timeline.invalidate('output');
});

// --- WAV download / upload / debug dump ---

els.downloadBtn.addEventListener('click', () => {
    if (!corrector) return;
    const samples = corrector.get_recording();
    if (samples.length === 0) return;
    downloadBlob(encodeWav(samples, sampleRate), 'recording.wav');
});

els.debugBtn.addEventListener('click', () => {
    if (!corrector) return;
    const dump = {
        sampleRate,
        input: Array.from(corrector.get_recording()),
        output: Array.from(corrector.get_output_recording()),
        targetContour: Array.from(targetContour),
        editedContour: editedContour ? Array.from(editedContour) : null,
    };
    downloadBlob(new Blob([JSON.stringify(dump)], { type: 'application/json' }), 'debug-dump.json');
});

els.uploadBtn.addEventListener('click', () => els.uploadInput.click());
els.uploadInput.addEventListener('change', (e) => {
    const input = /** @type {HTMLInputElement} */ (e.target);
    if (input.files[0]) uploadRecording(input.files[0]);
    input.value = '';
});

async function uploadRecording(file) {
    await init();
    newCorrector();
    applyScale();
    corrector.set_shift(parseFloat(slider.value));
    applyRetune();
    applyStrength();
    const buf = await file.arrayBuffer();
    const samples = decodeWav(buf);
    if (!samples || samples.length === 0) {
        els.status.textContent = 'Error: could not decode WAV';
        return;
    }
    applyHarmony();
    corrector.load_recording(samples);
    resetSession();
    corrector.clear_target_pitch_contour();

    // Process offline in chunks, yielding so the page stays responsive.
    const chunk = sampleRate; // ~1s of audio per slice
    for (let offset = 0; offset < samples.length;) {
        els.status.textContent = `Processing… ${Math.round((offset / samples.length) * 100)}%`;
        offset += corrector.process_offline(samples.subarray(offset), chunk);
        totalSamples = corrector.analyze();
        timeline.setTotal(totalSamples);
        await new Promise((r) => setTimeout(() => r(undefined), 0));
    }
    totalSamples = corrector.analyze();
    timeline.setTotal(totalSamples);
    targetContour = Array.from(corrector.take_target_pitch_contour());
    updateViewScales();
    timeline.fit();
    timeline.setPlayhead(0);
    corrector.seek(0);
    setState('stopped');
}

// --- Boot ---

// Browsers restore form-control values across reloads (the dropdown can
// say "Pitch" while the tracks still render the waveform default): adopt
// whatever the DOM says as the initial state instead of assuming defaults.
els.postCorrectionCb.checked = false; // session-bound; never restorable
updateShiftDisplay();
timeline.showTrackSelectors(splitCb.checked);
viewSelect.disabled = splitCb.checked;
if (!splitCb.checked) applySharedView();
updatePitchMerge();

// Gate on browser support, then pre-warm the audio engine.

(async function prewarm() {
    const missing = checkBrowserSupport();
    if (missing.length) {
        els.status.textContent =
            'This browser is missing ' +
            missing.join(', ') +
            '. Use a recent Chrome, Edge, or Firefox over HTTPS.';
        els.recordBtn.disabled = true;
        return;
    }
    els.recordBtn.disabled = true;
    els.status.textContent = 'Loading audio engine…';
    try {
        await init();
        warmup();
    } catch (e) {
        els.status.textContent =
            'Failed to load the audio engine (WASM). Check your connection and reload. (' + e + ')';
        return;
    }
    setState('idle');
})();
