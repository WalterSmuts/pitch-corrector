use crate::music::Interval;
use crate::music::Note;
use crate::music::Pitch;
use crate::music::Scale;
use crate::music::SimpleInterval;
use crate::pitch_correction::Harmonizer;
use crate::pitch_correction::HarmonyMode;
use crate::pitch_correction::PitchCorrector;
use crate::pitch_correction::PitchCorrectorControls;
use crate::session::PITCH_HOP;
use crate::session::PitchTrack;
use crate::session::SPEC_WINDOW;
use crate::session::SpectrogramRenderer;
use crate::session::waveform_peaks;
use crate::signal_processing::BUFFER_SIZE;
use crate::signal_processing::HOP_SIZE;
use crate::signal_processing::SpectralFreeze;
use crate::signal_processing::StreamProcessor;
use crate::track::Mirror;
use crate::track::Track;
use crate::track::spin_lock;
use crate::units::HopIdx;
use crate::units::SampleIdx;
use crossbeam_queue::ArrayQueue;
use crossbeam_utils::atomic::AtomicCell;

// Audio-thread cells must never hit AtomicCell's seqlock fallback.
const _: () = assert!(AtomicCell::<SampleIdx>::is_lock_free());
const _: () = assert!(AtomicCell::<f32>::is_lock_free());
use cpal::traits::DeviceTrait;
use cpal::traits::HostTrait;
use cpal::traits::StreamTrait;
use easyfft::dyn_size::realfft::DynRealFft;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::CanvasRenderingContext2d;
use web_sys::HtmlCanvasElement;
use web_sys::ImageData;

/// Prime the expensive, one-time initialization that would otherwise run on
/// the first `Record` click and jank the main thread (~0.3s): the FFT planner
/// caches (built lazily on first `real_fft`) and the phase-vocoder DSP buffers.
/// Call this once at page load, after `init()`, before the user records.
#[wasm_bindgen]
pub fn warmup() {
    // Prime the real-FFT planner caches for all sizes the app uses.
    let _ = vec![0.0f32; BUFFER_SIZE].real_fft();
    let _ = vec![0.0f32; SPEC_WINDOW].real_fft();
    // Run a few silent hops through a throwaway corrector so the phase
    // vocoder's plans/scratch and the OLA buffers are all allocated/primed.
    let mut corrector = PitchCorrector::new();
    for _ in 0..(BUFFER_SIZE * 4) {
        corrector.push_sample(0.0);
        while corrector.pop_sample().is_some() {}
    }
}

/// Lock a mutex shared with the audio threads without ever blocking on a
/// futex: `Mutex::lock` calls `Atomics.wait` under contention, which throws
/// on the wasm main thread. The audio callbacks hold these locks for
/// microseconds, so spinning on `try_lock` is safe and brief.
/// The DSP chain plus its control handle, independent of audio I/O.
struct PlaybackState {
    input_active: AtomicBool,
    /// The captured input lane (the source timeline).
    input: Track,
    /// The produced output lane (what re-processing overwrites).
    output: Track,
    playback_pos: AtomicCell<SampleIdx>,
    playing: AtomicBool,
    /// Set true the first time each worklet's audio callback runs, so the UI
    /// can wait until the whole pipeline (both render threads) is live before
    /// it starts drawing — avoids the async worklet setup janking mid-draw.
    input_started: AtomicBool,
    output_started: AtomicBool,
    /// Audible live passthrough while recording. Off by default: hearing
    /// yourself corrected is opt-in (and speakers feeding the mic loop).
    monitor: AtomicBool,
    /// True once the DSP pipeline has produced its first output sample
    /// since the last (re)start of a feed; underruns before that are the
    /// expected warm-up gap, not a problem worth logging.
    pipeline_primed: AtomicBool,
    /// Spectral-freeze audition (hold a key to sustain the frame under the
    /// playhead). The synth is installed by the UI thread and consumed by
    /// the output callback, which takes priority over the pipeline while
    /// `freeze_active` (with a short gain ramp so start/stop are
    /// click-free). Never recorded — it is an audition, not timeline data.
    freeze: Mutex<Option<SpectralFreeze>>,
    freeze_active: AtomicBool,
}

/// Main-thread analysis state: mirrors of the shared recordings plus
/// incremental pitch tracks and spectrogram scratch. The audio callbacks
/// append to the recordings with `try_lock` (dropping data on contention),
/// so all long-running work here happens on private mirrors — the shared
/// buffers are only locked briefly to copy out new samples.
struct Analysis {
    input: Mirror,
    output: Mirror,
    input_pitch: PitchTrack,
    /// Produced pitch per vocoder hop for every output voice
    /// ([main, 3rd, 5th, octave]), drained from the DSP's logs. The output
    /// pitch view plots these — never a pitch detector over the mixed
    /// output, which is polyphonic once harmonies are enabled.
    voice_pitch: [Vec<f32>; 4],
    /// Post-smoothing, full-strength aim of the main voice per vocoder hop
    /// (see `PitchCorrectorControls::aim_pitch_log`).
    aim_pitch: Vec<f32>,
    /// Snap target (nearest scale note, or the installed contour entry) per
    /// vocoder hop (Hz; 0 = unvoiced). The single consumer of the DSP's
    /// target log — the UI reads this mirror both live and at stop (it
    /// seeds the editable contour).
    target_pitch: Vec<f32>,
    spec: SpectrogramRenderer,
    rgba: Vec<u8>,
}

impl Analysis {
    fn new(sample_rate: f32) -> Self {
        Analysis {
            input: Mirror::new(),
            output: Mirror::new(),
            input_pitch: PitchTrack::new(sample_rate),
            voice_pitch: std::array::from_fn(|_| Vec::new()),
            aim_pitch: Vec::new(),
            target_pitch: Vec::new(),
            spec: SpectrogramRenderer::new(),
            rgba: Vec::new(),
        }
    }

    /// Pull new samples from the shared recordings (short locks), then
    /// advance pitch analysis on the mirrors. A shrunken output recording
    /// (playback overwriting from the seek position) invalidates and
    /// re-analyzes the tail; a shrunken input recording resets the mirror.
    fn sync(&mut self, playback: &PlaybackState, controls: &PitchCorrectorControls) {
        if self.input.catch_up(&playback.input).is_some() {
            // Input rewrites are whole-session resets; restart the tracker
            // (the mirror already discarded its stale tail).
            self.input_pitch.reset();
        }
        // Drain the DSP's per-voice pitch logs first: entries produced
        // before a playback-seek truncation belong before it on the
        // timeline, so append-then-truncate keeps them aligned.
        for (voice, track) in self.voice_pitch.iter_mut().enumerate() {
            controls.drain_voice_pitch(voice, track);
        }
        controls.drain_aim_pitch(&mut self.aim_pitch);
        self.target_pitch.extend(
            controls
                .take_target_pitch_contour()
                .iter()
                .map(|p| p.map_or(0.0, |p| p.to_freq())),
        );
        if let Some(len) = self.output.catch_up(&playback.output) {
            // Output rewrite (playback seek): drop derived pitch past it.
            let hops: HopIdx<HOP_SIZE> = HopIdx::containing(SampleIdx(len));
            for track in &mut self.voice_pitch {
                track.truncate(hops.0);
            }
            self.aim_pitch.truncate(hops.0);
            self.target_pitch.truncate(hops.0);
        }
        self.input_pitch.analyze(self.input.samples());
    }

    fn reset(&mut self, playback: &PlaybackState, controls: &PitchCorrectorControls) {
        // Adopt the tracks' current state (consuming any pending rewrite
        // marks): entries logged after this reset must not be truncated by
        // a rewrite that happened before it.
        self.input.resync(&playback.input);
        self.output.resync(&playback.output);
        self.input_pitch.reset();
        for (voice, track) in self.voice_pitch.iter_mut().enumerate() {
            track.clear();
            // Discard any stale log entries from the previous session.
            controls.drain_voice_pitch(voice, track);
            track.clear();
        }
        self.aim_pitch.clear();
        controls.drain_aim_pitch(&mut self.aim_pitch);
        self.aim_pitch.clear();
        self.target_pitch.clear();
        controls.clear_target_pitch_contour();
    }
}

#[wasm_bindgen]
pub struct WebPitchCorrector {
    input_stream: cpal::Stream,
    output_stream: cpal::Stream,
    processor: Arc<Mutex<Harmonizer>>,
    controls: Arc<PitchCorrectorControls>,
    playback: Arc<PlaybackState>,
    analysis: Mutex<Analysis>,
    sample_rate: f32,
}

#[wasm_bindgen]
impl WebPitchCorrector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebPitchCorrector, JsValue> {
        console_log::init_with_level(log::Level::Info).ok();

        // With the `worklet` feature, run audio on the AudioWorklet host so the
        // DSP executes on the audio render thread instead of the main thread
        // (see the web performance notes). Otherwise use the default host
        // (ScriptProcessorNode on the main thread).
        #[cfg(feature = "worklet")]
        let host = cpal::host_from_id(cpal::HostId::AudioWorklet)
            .map_err(|e| JsValue::from_str(&format!("AudioWorklet host unavailable: {e:?}")))?;
        #[cfg(not(feature = "worklet"))]
        let host = cpal::default_host();

        let input_device = host
            .default_input_device()
            .ok_or_else(|| JsValue::from_str("No input device"))?;
        let output_device = host
            .default_output_device()
            .ok_or_else(|| JsValue::from_str("No output device"))?;

        let input_config = input_device
            .default_input_config()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        let output_config = output_device
            .default_output_config()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        // Detection must use the real device rate, not the native default.
        let sample_rate = input_config.sample_rate() as f32;
        // The web hosts default to stereo and deliver interleaved frames.
        // Everything internal is mono: mix input frames down and fan the
        // output sample out to every channel. Treating interleaved stereo
        // as mono doubles the sample count and halves every frequency.
        let input_channels = input_config.channels() as usize;
        let output_channels = output_config.channels() as usize;

        // The DSP pipeline has single-owner (`&mut`) semantics; no locks
        // inside. It has exactly two *serialized* drivers — the output
        // callback (live) and process_offline (upload, streams paused) —
        // expressed as one boundary Mutex that is uncontended by
        // construction. The input callback never touches it: mic samples
        // cross to the driver through a lock-free ring.
        let processor = Arc::new(Mutex::new(Harmonizer::with_sample_rate(sample_rate)));
        let controls = processor.lock().unwrap().controls();
        let mic_ring = Arc::new(ArrayQueue::<f32>::new(BUFFER_SIZE * 8));
        let mic_ring_in = mic_ring.clone();
        let output_processor = processor.clone();

        let playback = Arc::new(PlaybackState {
            input_active: AtomicBool::new(true),
            input: Track::new(),
            output: Track::new(),
            playback_pos: AtomicCell::new(SampleIdx(0)),
            playing: AtomicBool::new(false),
            input_started: AtomicBool::new(false),
            output_started: AtomicBool::new(false),
            monitor: AtomicBool::new(false),
            pipeline_primed: AtomicBool::new(false),
            freeze: Mutex::new(None),
            freeze_active: AtomicBool::new(false),
        });

        let input_playback = playback.clone();
        let input_stream = input_device
            .build_input_stream(
                input_config.into(),
                move |data: &[f32], _| {
                    input_playback.input_started.store(true, Ordering::Relaxed);
                    if !input_playback.input_active.load(Ordering::Relaxed) {
                        return;
                    }
                    let mut rec = input_playback.input.writer();
                    for frame in data.chunks_exact(input_channels) {
                        let s = frame.iter().sum::<f32>() / input_channels as f32;
                        if mic_ring_in.push(s).is_err() {
                            log::warn!("Input callback: mic ring overflow — dropping sample");
                        }
                        if let Some(rec) = rec.as_mut() {
                            rec.push(s);
                        }
                    }
                },
                |err| log::error!("Input error: {}", err),
                None,
            )
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        let output_playback = playback.clone();
        // Freeze-audition gain (f32 bits) and its ~5ms one-pole coefficient,
        // owned by the output callback.
        let freeze_gain = Arc::new(AtomicCell::new(0.0f32));
        let freeze_gain_alpha = 1.0 - (-1.0 / (0.005 * sample_rate)).exp();
        // Underruns are expected while the pipeline warms up (the vocoder
        // needs a full window of input before it emits anything) and when
        // nothing is feeding the pipeline; only warn about gaps after the
        // pipeline has produced output, while actively fed — and once per
        // callback, not once per sample (a starved stream would otherwise
        // log tens of thousands of lines per second).
        let output_stream = output_device
            .build_output_stream(
                output_config.into(),
                move |data: &mut [f32], _| {
                    output_playback
                        .output_started
                        .store(true, Ordering::Relaxed);

                    // Freeze audition: synthesize directly, bypassing the
                    // pipeline. Only reachable when neither recording nor
                    // playing (the UI gates it), so nothing else needs the
                    // buffer. Gain ramps ~5ms both ways; after the release
                    // ramp finishes we fall through to the normal path.
                    {
                        let active = output_playback.freeze_active.load(Ordering::Relaxed);
                        let mut gain = freeze_gain.load();
                        if (active || gain > 1e-4)
                            && let Ok(mut fz) = output_playback.freeze.try_lock()
                            && let Some(fz) = fz.as_mut()
                        {
                            let target = if active { 1.0 } else { 0.0 };
                            for frame in data.chunks_exact_mut(output_channels) {
                                gain += freeze_gain_alpha * (target - gain);
                                frame.fill(fz.next_sample() * gain);
                            }
                            freeze_gain.store(gain);
                            return;
                        }
                    }

                    // Single boundary lock per callback; only contended
                    // while process_offline runs, and the streams are
                    // paused then — treat it as an underrun if it ever is.
                    let Ok(mut processor) = output_processor.try_lock() else {
                        for frame in data.chunks_exact_mut(output_channels) {
                            frame.fill(0.0);
                        }
                        return;
                    };
                    let fed = output_playback.playing.load(Ordering::Relaxed)
                        || output_playback.input_active.load(Ordering::Relaxed);
                    let frames = data.len() / output_channels;
                    if output_playback.input_active.load(Ordering::Relaxed) {
                        // Live: drain captured mic samples into the pipeline.
                        while let Some(s) = mic_ring.pop() {
                            processor.push_sample(s);
                        }
                    } else {
                        // Not recording: discard any stale capture residue so
                        // it can't pollute the next playback re-process.
                        while mic_ring.pop().is_some() {}
                    }
                    if output_playback.playing.load(Ordering::Relaxed)
                        && let Some(rec) = output_playback.input.writer()
                    {
                        let mut p = output_playback.playback_pos.load().0;
                        for _ in 0..frames {
                            if p < rec.len() {
                                processor.push_sample(rec[p]);
                                p += 1;
                            }
                        }
                        if p >= rec.len() {
                            output_playback.playing.store(false, Ordering::Relaxed);
                        }
                        output_playback.playback_pos.store(SampleIdx(p));
                    }
                    // While recording live, route audio to the speakers only
                    // with passthrough on; the pipeline still runs and the
                    // output is still captured for the visuals either way.
                    let live_muted = output_playback.input_active.load(Ordering::Relaxed)
                        && !output_playback.monitor.load(Ordering::Relaxed);
                    let mut missed = 0usize;
                    for frame in data.chunks_exact_mut(output_channels) {
                        match processor.pop_sample() {
                            Some(s) => {
                                output_playback
                                    .pipeline_primed
                                    .store(true, Ordering::Relaxed);
                                frame.fill(if live_muted { 0.0 } else { s });
                                // Capture the produced audio both while
                                // recording live and while re-processing
                                // during playback (playback truncates the
                                // buffer to the play position first, so
                                // appends stay timeline-aligned).
                                if (output_playback.input_active.load(Ordering::Relaxed)
                                    || output_playback.playing.load(Ordering::Relaxed))
                                    && let Some(mut rec) = output_playback.output.writer()
                                {
                                    rec.push(s);
                                }
                            }
                            None => {
                                missed += 1;
                                frame.fill(0.0);
                            }
                        }
                    }
                    if missed > 0 && fed && output_playback.pipeline_primed.load(Ordering::Relaxed)
                    {
                        log::warn!("Output callback: underrun — inserted {missed} silent samples");
                    }
                },
                |err| log::error!("Output error: {}", err),
                None,
            )
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        input_stream
            .play()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        output_stream
            .play()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        Ok(WebPitchCorrector {
            input_stream,
            output_stream,
            processor,
            controls,
            playback,
            analysis: Mutex::new(Analysis::new(sample_rate)),
            sample_rate,
        })
    }

    /// The actual device sample rate (Hz) the streams run at. JS reads this
    /// as the single source of truth (e.g. for WAV export) instead of
    /// hardcoding a rate that may not match the browser AudioContext.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// True once both the input and output worklets have run their first
    /// callback, i.e. the whole audio pipeline is live. The UI waits for this
    /// before it starts drawing so the async worklet setup can't jank the
    /// draw loop (and audio is already flowing).
    pub fn is_audio_ready(&self) -> bool {
        self.playback.input_started.load(Ordering::Relaxed)
            && self.playback.output_started.load(Ordering::Relaxed)
    }

    pub fn set_shift(&self, semitones: f32) {
        let s = semitones.round() as i32;
        let octaves = s.div_euclid(12) as i8;
        let simple = SimpleInterval::ALL[s.rem_euclid(12) as usize];
        self.controls.set_shift(Interval::compound(simple, octaves));
    }

    pub fn get_shift(&self) -> f32 {
        self.controls.get_shift().semitones() as f32
    }

    /// Returns the recorded target pitch contour and clears it. One entry
    /// per vocoder hop: entry `i` belongs at input sample
    /// `i * vocoder_hop()`. Never rescale it against the recording length —
    /// the underlying log is bounded and may be shorter than the recording
    /// (missing tail), but every entry it does have is hop-true.
    /// Snap-target track per vocoder hop (Hz; 0 = unvoiced), available live
    /// and at stop. Replaces the old destructive queue drain at stop — the
    /// analysis mirror is the single consumer of the DSP's target log.
    pub fn target_pitch_track(&self) -> Vec<f32> {
        self.analysis.lock().unwrap().target_pitch.clone()
    }

    pub fn clear_target_pitch_contour(&self) {
        self.controls.clear_target_pitch_contour();
    }

    pub fn set_scale(&self, bits: u16) {
        self.controls.set_scale(Scale::from_bits(bits));
    }

    pub fn get_scale(&self) -> u16 {
        self.controls.get_scale().bits()
    }

    pub fn stop(&self) {
        self.playback.input_active.store(false, Ordering::Relaxed);
        let _ = self.input_stream.pause();
        let _ = self.output_stream.pause();
    }

    /// Begin a fresh recording on the existing audio graph. Reusing the
    /// corrector (instead of constructing a new one per take) matters:
    /// cpal's web hosts don't reliably stop a paused stream's render
    /// callbacks, so a second live AudioContext leaves the old one fighting
    /// over the capture device ("closure invoked after being dropped"
    /// errors, duplicated sample delivery).
    pub fn start_recording(&self) -> Result<(), JsValue> {
        self.playback.playing.store(false, Ordering::Relaxed);
        self.playback.input.clear();
        self.playback.output.clear();
        self.playback.playback_pos.store(SampleIdx(0));
        self.analysis
            .lock()
            .unwrap()
            .reset(&self.playback, &self.controls);
        self.controls.clear_target_pitch_contour();
        self.controls.clear_contour();
        self.playback
            .pipeline_primed
            .store(false, Ordering::Relaxed);
        self.playback.input_active.store(true, Ordering::Relaxed);
        self.input_stream
            .play()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        self.output_stream
            .play()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        Ok(())
    }

    /// Enabled harmony voices: bit0 = 3rd, bit1 = 5th, bit2 = octave.
    pub fn set_harmony(&self, mask: u8) {
        self.controls.set_harmony(mask);
    }

    /// Diatonic (walk the selected scale) vs absolute (fixed semitones).
    /// (Bool at the wasm boundary; typed HarmonyMode inside.)
    pub fn set_harmony_in_key(&self, in_key: bool) {
        self.controls.set_harmony_mode(if in_key {
            HarmonyMode::InKey
        } else {
            HarmonyMode::Absolute
        });
    }

    /// Dry bypass (A/B): hear the uncorrected voice. Applies live, during
    /// both recording and playback re-processing.
    pub fn set_bypass(&self, on: bool) {
        self.controls.set_bypass(on);
    }

    /// Retune speed in milliseconds (time constant of the correction
    /// smoothing). Small = hard snap, large = transparent glide.
    pub fn set_retune_speed_ms(&self, ms: f32) {
        self.controls.set_retune_tau_seconds(ms / 1000.0);
    }

    /// Correction strength 0..=1: fraction of the correction interval
    /// applied. Manual shift and harmonies are unaffected.
    pub fn set_strength(&self, strength: f32) {
        self.controls.set_strength(strength);
    }

    /// Sustain the sound at `position` (samples on the output timeline)
    /// using a spectral freeze — hold-a-key audition. Only meaningful while
    /// stopped or paused; returns false if that doesn't hold or there is
    /// not enough audio around the position.
    pub fn start_freeze(&self, position: f64) -> bool {
        if self.playback.playing.load(Ordering::Relaxed)
            || self.playback.input_active.load(Ordering::Relaxed)
        {
            return false;
        }
        let a = self.analysis.lock().unwrap();
        // Analyze the window *ending* at the cursor — the audio just heard.
        // During/after playback the output only exists up to the playhead
        // (minus pipeline latency), so clamp to what is actually there and
        // a window centered on the cursor would have no second half anyway.
        let end = (position.max(0.0) as usize).min(a.output.len());
        let center = SampleIdx(end.saturating_sub((BUFFER_SIZE + HOP_SIZE) / 2));
        match SpectralFreeze::new(a.output.samples(), center) {
            Some(fz) => {
                *spin_lock(&self.playback.freeze) = Some(fz);
                self.playback.freeze_active.store(true, Ordering::Relaxed);
                let _ = self.output_stream.play();
                true
            }
            None => false,
        }
    }

    /// Release the freeze; the callback ramps it out click-free.
    pub fn stop_freeze(&self) {
        self.playback.freeze_active.store(false, Ordering::Relaxed);
    }

    pub fn is_freezing(&self) -> bool {
        self.playback.freeze_active.load(Ordering::Relaxed)
    }

    /// Audible live passthrough while recording (default off). Playback is
    /// always audible; this only gates what reaches the speakers live.
    pub fn set_monitor(&self, on: bool) {
        self.playback.monitor.store(on, Ordering::Relaxed);
    }

    pub fn recording_len(&self) -> usize {
        self.playback.input.len()
    }

    pub fn get_recording(&self) -> Vec<f32> {
        self.playback.input.snapshot()
    }

    pub fn load_recording(&self, samples: &[f32]) {
        self.playback.input.replace(samples);
        self.playback.output.clear();
        self.playback.playback_pos.store(SampleIdx(0));
        self.playback.input_active.store(false, Ordering::Relaxed);
        self.analysis
            .lock()
            .unwrap()
            .reset(&self.playback, &self.controls);
        let _ = self.input_stream.pause();
        let _ = self.output_stream.pause();
    }

    pub fn process_offline(&self, samples: &[f32], count: usize) -> usize {
        // Streams are paused here (load_recording), so this never contends
        // with the output callback.
        let mut processor = spin_lock(&self.processor);
        let n = samples.len().min(count);
        for &s in &samples[..n] {
            processor.push_sample(s);
            while let Some(o) = processor.pop_sample() {
                self.playback.output.locked().push(o);
            }
        }
        n
    }

    pub fn get_output_recording(&self) -> Vec<f32> {
        self.playback.output.snapshot()
    }

    pub fn play_recording(&self) -> Result<(), JsValue> {
        if self.playback.input.is_empty() {
            return Ok(());
        }
        self.playback.input_active.store(false, Ordering::Relaxed);
        // Re-processed output overwrites the timeline from the play position:
        // drop everything after it (or pad silence up to it) so the
        // callback's appends line up.
        let pos = self.playback.playback_pos.load().0;
        self.playback.output.rewrite_from(pos);
        // The edited contour lives on the absolute hop timeline; align its
        // cursor with where playback starts.
        self.controls
            .seek_contour(HopIdx::containing(SampleIdx(pos)));
        self.playback
            .pipeline_primed
            .store(false, Ordering::Relaxed);
        self.playback.playing.store(true, Ordering::Relaxed);
        let _ = self.output_stream.play();
        Ok(())
    }

    pub fn stop_playback(&self) {
        self.playback.playing.store(false, Ordering::Relaxed);
        let _ = self.output_stream.pause();
    }

    pub fn is_playing(&self) -> bool {
        self.playback.playing.load(Ordering::Relaxed)
    }

    pub fn playback_progress(&self) -> f32 {
        let len = self.recording_len();
        if len == 0 {
            return 0.0;
        }
        self.playback.playback_pos.load().0 as f32 / len as f32
    }

    pub fn seek(&self, fraction: f32) {
        let len = self.recording_len() as f32;
        let pos = (fraction.clamp(0.0, 1.0) * len) as usize;
        self.playback.playback_pos.store(SampleIdx(pos));
        // If we're mid-playback, the output write head must jump with us.
        if self.playback.playing.load(Ordering::Relaxed) {
            self.playback.output.rewrite_from(pos);
            self.controls
                .seek_contour(HopIdx::containing(SampleIdx(pos)));
        }
    }

    pub fn scale_bits(preset: &str, root: u8) -> u16 {
        let root_note = Note::ALL[root as usize % 12];
        match preset {
            "off" => Scale::empty().bits(),
            "chromatic" => Scale::chromatic().bits(),
            "major" => Scale::major(root_note).bits(),
            "minor" => Scale::minor(root_note).bits(),
            "pentatonic" => Scale::pentatonic(root_note).bits(),
            _ => Scale::chromatic().bits(),
        }
    }

    pub fn snap_to_scale(freq: f32, note_bits: u16) -> f32 {
        let notes = Scale::from_bits(note_bits);
        notes.nearest_pitch(freq).to_freq()
    }

    /// Set a pitch contour as the active target for playback.
    pub fn set_contour(&self, contour: &[f32]) {
        let pitches = contour
            .iter()
            .map(|&f| {
                if f > 0.0 {
                    Some(Pitch::from_freq(f))
                } else {
                    None
                }
            })
            .collect();
        self.controls.set_contour(pitches);
    }

    /// Restore the default NoteSnapper target.
    pub fn clear_contour(&self) {
        self.controls.clear_contour();
    }

    // --- Session data APIs (timeline UI renders from these) ---

    /// Samples per pitch-track hop.
    pub fn pitch_hop(&self) -> u32 {
        PITCH_HOP as u32
    }

    /// Pull new audio into the analysis mirrors and advance the pitch
    /// tracks. Call once per animation frame; incremental and cheap when
    /// nothing new arrived. Returns the input length in samples.
    pub fn analyze(&self) -> f64 {
        let mut a = self.analysis.lock().unwrap();
        a.sync(&self.playback, &self.controls);
        a.input.len() as f64
    }

    /// Analyzed output length in samples. Lags the input by the pipeline
    /// latency during recording; the UI uses it as the repaint watermark
    /// for output tracks.
    pub fn output_len(&self) -> f64 {
        self.analysis.lock().unwrap().output.len() as f64
    }

    /// Detected input pitch per hop (Hz, 0 = unvoiced).
    pub fn input_pitch_track(&self) -> Vec<f32> {
        self.analysis.lock().unwrap().input_pitch.track().to_vec()
    }

    /// Samples per vocoder hop (the granularity of the voice pitch logs).
    pub fn vocoder_hop(&self) -> u32 {
        HOP_SIZE as u32
    }

    /// Produced pitch of the main corrected voice per vocoder hop (Hz,
    /// 0 = unvoiced), logged by the DSP itself — no detector runs on the
    /// (possibly polyphonic) mixed output.
    pub fn output_pitch_track(&self) -> Vec<f32> {
        self.analysis.lock().unwrap().voice_pitch[0].clone()
    }

    /// Post-smoothing, full-strength aim of the main voice per vocoder hop
    /// (Hz; 0 = unvoiced). Where the output would land at strength 1; equal
    /// to `output_pitch_track` when strength is 1.
    pub fn aim_pitch_track(&self) -> Vec<f32> {
        self.analysis.lock().unwrap().aim_pitch.clone()
    }

    /// Produced pitch of harmony voice 1..=3 (3rd, 5th, octave) per vocoder
    /// hop (Hz, 0 = silent/disabled).
    pub fn harmony_pitch_track(&self, voice: u32) -> Vec<f32> {
        match voice {
            1..=3 => self.analysis.lock().unwrap().voice_pitch[voice as usize].clone(),
            _ => Vec::new(),
        }
    }

    /// Min/max waveform peaks for a sample range of the input, binned to
    /// `bins` columns; `bins * 2` values interleaved `[min, max]`.
    pub fn input_peaks(&self, start_sample: f64, end_sample: f64, bins: u32) -> Vec<f32> {
        let a = self.analysis.lock().unwrap();
        waveform_peaks(a.input.samples(), start_sample, end_sample, bins as usize)
    }

    /// Same as `input_peaks`, for the produced output.
    pub fn output_peaks(&self, start_sample: f64, end_sample: f64, bins: u32) -> Vec<f32> {
        let a = self.analysis.lock().unwrap();
        waveform_peaks(a.output.samples(), start_sample, end_sample, bins as usize)
    }

    /// Render `width_px` spectrogram columns of the input recording into
    /// `canvas` at `dest_x`, covering the time range starting at
    /// `start_sample` with `samples_per_px` samples per column.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_input_spectrogram_range(
        &self,
        canvas: &HtmlCanvasElement,
        dest_x: u32,
        width_px: u32,
        start_sample: f64,
        samples_per_px: f64,
        f_lo: f32,
        f_hi: f32,
    ) -> Result<(), JsValue> {
        self.draw_spec_range(
            canvas,
            dest_x,
            width_px,
            start_sample,
            samples_per_px,
            f_lo,
            f_hi,
            true,
        )
    }

    /// Same as `draw_input_spectrogram_range`, for the produced output.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_output_spectrogram_range(
        &self,
        canvas: &HtmlCanvasElement,
        dest_x: u32,
        width_px: u32,
        start_sample: f64,
        samples_per_px: f64,
        f_lo: f32,
        f_hi: f32,
    ) -> Result<(), JsValue> {
        self.draw_spec_range(
            canvas,
            dest_x,
            width_px,
            start_sample,
            samples_per_px,
            f_lo,
            f_hi,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_spec_range(
        &self,
        canvas: &HtmlCanvasElement,
        dest_x: u32,
        width_px: u32,
        start_sample: f64,
        samples_per_px: f64,
        f_lo: f32,
        f_hi: f32,
        input: bool,
    ) -> Result<(), JsValue> {
        if width_px == 0 {
            return Ok(());
        }
        let height = canvas.height();
        let a = &mut *self.analysis.lock().unwrap();
        let samples = if input {
            a.input.samples()
        } else {
            a.output.samples()
        };
        a.spec.render(
            samples,
            start_sample,
            samples_per_px,
            width_px as usize,
            height as usize,
            f_lo,
            f_hi,
            &mut a.rgba,
        );
        // ImageData rejects views into SharedArrayBuffer-backed wasm memory
        // (the threads build), so copy into a fresh, unshared JS array.
        let data = js_sys::Uint8ClampedArray::new_with_length(a.rgba.len() as u32);
        data.copy_from(&a.rgba);
        let img = ImageData::new_with_js_u8_clamped_array_and_sh(&data, width_px, height)?;
        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("no 2d context"))?
            .dyn_into()?;
        ctx.put_image_data(&img, dest_x as f64, 0.0)
    }
}
