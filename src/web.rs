use crate::music::{Interval, Note, Pitch, Scale, SimpleInterval};
use crate::pitch_correction::{PitchCorrector, PitchCorrectorControls};
use crate::session::{waveform_peaks, PitchTrack, SpectrogramRenderer, PITCH_HOP, SPEC_WINDOW};
use crate::signal_processing::{StreamProcessor, BUFFER_SIZE};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use easyfft::dyn_size::realfft::DynRealFft;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

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
    let corrector = PitchCorrector::new();
    for _ in 0..(BUFFER_SIZE * 4) {
        corrector.push_sample(0.0);
        while corrector.pop_sample().is_some() {}
    }
}

/// Lock a mutex shared with the audio threads without ever blocking on a
/// futex: `Mutex::lock` calls `Atomics.wait` under contention, which throws
/// on the wasm main thread. The audio callbacks hold these locks for
/// microseconds, so spinning on `try_lock` is safe and brief.
fn spin_lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    loop {
        match m.try_lock() {
            Ok(g) => return g,
            Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
            Err(std::sync::TryLockError::Poisoned(p)) => return p.into_inner(),
        }
    }
}

/// The DSP chain plus its control handle, independent of audio I/O.
pub struct Pipeline {
    processor: Arc<dyn StreamProcessor + Send + Sync>,
    controls: Arc<PitchCorrectorControls>,
}

impl Pipeline {
    pub fn new(sample_rate: f32) -> Self {
        let corrector = PitchCorrector::with_sample_rate(sample_rate);
        let controls = corrector.controls();
        Pipeline {
            processor: Arc::new(corrector),
            controls,
        }
    }

    pub fn processor(&self) -> &Arc<dyn StreamProcessor + Send + Sync> {
        &self.processor
    }

    pub fn controls(&self) -> &Arc<PitchCorrectorControls> {
        &self.controls
    }
}

struct PlaybackState {
    input_active: AtomicBool,
    recording: Mutex<Vec<f32>>,
    output_recording: Mutex<Vec<f32>>,
    playback_pos: AtomicU32,
    playing: AtomicBool,
    /// Set true the first time each worklet's audio callback runs, so the UI
    /// can wait until the whole pipeline (both render threads) is live before
    /// it starts drawing — avoids the async worklet setup janking mid-draw.
    input_started: AtomicBool,
    output_started: AtomicBool,
    /// True once the DSP pipeline has produced its first output sample
    /// since the last (re)start of a feed; underruns before that are the
    /// expected warm-up gap, not a problem worth logging.
    pipeline_primed: AtomicBool,
}

/// Main-thread analysis state: mirrors of the shared recordings plus
/// incremental pitch tracks and spectrogram scratch. The audio callbacks
/// append to the recordings with `try_lock` (dropping data on contention),
/// so all long-running work here happens on private mirrors — the shared
/// buffers are only locked briefly to copy out new samples.
struct Analysis {
    input_samples: Vec<f32>,
    output_samples: Vec<f32>,
    input_pitch: PitchTrack,
    output_pitch: PitchTrack,
    spec: SpectrogramRenderer,
    rgba: Vec<u8>,
}

impl Analysis {
    fn new(sample_rate: f32) -> Self {
        Analysis {
            input_samples: Vec::new(),
            output_samples: Vec::new(),
            input_pitch: PitchTrack::new(sample_rate),
            output_pitch: PitchTrack::new(sample_rate),
            spec: SpectrogramRenderer::new(),
            rgba: Vec::new(),
        }
    }

    /// Pull new samples from the shared recordings (short locks), then
    /// advance pitch analysis on the mirrors. A shrunken output recording
    /// (playback overwriting from the seek position) invalidates and
    /// re-analyzes the tail; a shrunken input recording resets the mirror.
    fn sync(&mut self, playback: &PlaybackState) {
        {
            let rec = spin_lock(&playback.recording);
            if rec.len() < self.input_samples.len() {
                self.input_samples.clear();
                self.input_pitch.reset();
            }
            let n = self.input_samples.len();
            self.input_samples.extend_from_slice(&rec[n..]);
        }
        {
            let out = spin_lock(&playback.output_recording);
            if out.len() < self.output_samples.len() {
                self.output_samples.truncate(out.len());
                self.output_pitch.invalidate_from(out.len());
            }
            let n = self.output_samples.len();
            self.output_samples.extend_from_slice(&out[n..]);
        }
        self.input_pitch.analyze(&self.input_samples);
        self.output_pitch.analyze(&self.output_samples);
    }

    fn reset(&mut self) {
        self.input_samples.clear();
        self.output_samples.clear();
        self.input_pitch.reset();
        self.output_pitch.reset();
    }
}

#[wasm_bindgen]
pub struct WebPitchCorrector {
    input_stream: cpal::Stream,
    output_stream: cpal::Stream,
    pipeline: Pipeline,
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

        let pipeline = Pipeline::new(sample_rate);
        let input_processor = pipeline.processor().clone();
        let output_processor = pipeline.processor().clone();

        let playback = Arc::new(PlaybackState {
            input_active: AtomicBool::new(true),
            recording: Mutex::new(Vec::new()),
            output_recording: Mutex::new(Vec::new()),
            playback_pos: AtomicU32::new(0),
            playing: AtomicBool::new(false),
            input_started: AtomicBool::new(false),
            output_started: AtomicBool::new(false),
            pipeline_primed: AtomicBool::new(false),
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
                    if let Ok(mut rec) = input_playback.recording.try_lock() {
                        rec.extend_from_slice(data);
                    }
                    for &sample in data {
                        input_processor.push_sample(sample);
                    }
                },
                |err| log::error!("Input error: {}", err),
                None,
            )
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        let output_playback = playback.clone();
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
                    let fed = output_playback.playing.load(Ordering::Relaxed)
                        || output_playback.input_active.load(Ordering::Relaxed);
                    if output_playback.playing.load(Ordering::Relaxed) {
                        if let Ok(rec) = output_playback.recording.try_lock() {
                            let mut p =
                                output_playback.playback_pos.load(Ordering::Relaxed) as usize;
                            for _ in 0..data.len() {
                                if p < rec.len() {
                                    output_processor.push_sample(rec[p]);
                                    p += 1;
                                }
                            }
                            if p >= rec.len() {
                                output_playback.playing.store(false, Ordering::Relaxed);
                            }
                            output_playback
                                .playback_pos
                                .store(p as u32, Ordering::Relaxed);
                        }
                    }
                    let mut missed = 0usize;
                    for sample in data.iter_mut() {
                        match output_processor.pop_sample() {
                            Some(s) => {
                                output_playback
                                    .pipeline_primed
                                    .store(true, Ordering::Relaxed);
                                *sample = s;
                                // Capture the produced audio both while
                                // recording live and while re-processing
                                // during playback (playback truncates the
                                // buffer to the play position first, so
                                // appends stay timeline-aligned).
                                if output_playback.input_active.load(Ordering::Relaxed)
                                    || output_playback.playing.load(Ordering::Relaxed)
                                {
                                    if let Ok(mut rec) = output_playback.output_recording.try_lock()
                                    {
                                        rec.push(s);
                                    }
                                }
                            }
                            None => {
                                missed += 1;
                                *sample = 0.0;
                            }
                        }
                    }
                    if missed > 0
                        && fed
                        && output_playback.pipeline_primed.load(Ordering::Relaxed)
                    {
                        log::warn!(
                            "Output callback: underrun — inserted {missed} silent samples"
                        );
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
            pipeline,
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
        self.pipeline
            .controls
            .set_shift(Interval::compound(simple, octaves));
    }

    pub fn get_shift(&self) -> f32 {
        self.pipeline.controls.get_shift().semitones() as f32
    }

    /// Returns the recorded target pitch contour (one entry per phase vocoder hop)
    /// and clears it.
    pub fn take_target_pitch_contour(&self) -> Vec<f32> {
        self.pipeline
            .controls
            .take_target_pitch_contour()
            .iter()
            .map(|p| p.map_or(0.0, |p| p.to_freq()))
            .collect()
    }

    pub fn clear_target_pitch_contour(&self) {
        self.pipeline.controls.clear_target_pitch_contour();
    }

    pub fn set_scale(&self, bits: u16) {
        self.pipeline.controls.set_scale(Scale::from_bits(bits));
    }

    pub fn get_scale(&self) -> u16 {
        self.pipeline.controls.get_scale().bits()
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
        spin_lock(&self.playback.recording).clear();
        spin_lock(&self.playback.output_recording).clear();
        self.playback.playback_pos.store(0, Ordering::Relaxed);
        self.analysis.lock().unwrap().reset();
        self.pipeline.controls.clear_target_pitch_contour();
        self.pipeline.controls.clear_contour();
        self.playback.pipeline_primed.store(false, Ordering::Relaxed);
        self.playback.input_active.store(true, Ordering::Relaxed);
        self.input_stream
            .play()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        self.output_stream
            .play()
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        Ok(())
    }

    pub fn recording_len(&self) -> usize {
        spin_lock(&self.playback.recording).len()
    }

    pub fn get_recording(&self) -> Vec<f32> {
        spin_lock(&self.playback.recording).clone()
    }

    pub fn load_recording(&self, samples: &[f32]) {
        *spin_lock(&self.playback.recording) = samples.to_vec();
        spin_lock(&self.playback.output_recording).clear();
        self.playback.playback_pos.store(0, Ordering::Relaxed);
        self.playback.input_active.store(false, Ordering::Relaxed);
        self.analysis.lock().unwrap().reset();
        let _ = self.input_stream.pause();
        let _ = self.output_stream.pause();
    }

    pub fn process_offline(&self, samples: &[f32], count: usize) -> usize {
        let n = samples.len().min(count);
        for &s in &samples[..n] {
            self.pipeline.processor.push_sample(s);
            while let Some(o) = self.pipeline.processor.pop_sample() {
                spin_lock(&self.playback.output_recording).push(o);
            }
        }
        n
    }

    pub fn get_output_recording(&self) -> Vec<f32> {
        spin_lock(&self.playback.output_recording).clone()
    }

    pub fn play_recording(&self) -> Result<(), JsValue> {
        if spin_lock(&self.playback.recording).is_empty() {
            return Ok(());
        }
        self.playback.input_active.store(false, Ordering::Relaxed);
        // Re-processed output overwrites the timeline from the play position:
        // drop everything after it (or pad silence up to it) so the
        // callback's appends line up.
        let pos = self.playback.playback_pos.load(Ordering::Relaxed) as usize;
        {
            let mut out = spin_lock(&self.playback.output_recording);
            out.truncate(pos);
            out.resize(pos, 0.0);
        }
        self.playback.pipeline_primed.store(false, Ordering::Relaxed);
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
        self.playback.playback_pos.load(Ordering::Relaxed) as f32 / len as f32
    }

    pub fn seek(&self, fraction: f32) {
        let len = self.recording_len() as f32;
        let pos = (fraction.clamp(0.0, 1.0) * len) as u32;
        self.playback.playback_pos.store(pos, Ordering::Relaxed);
        // If we're mid-playback, the output write head must jump with us.
        if self.playback.playing.load(Ordering::Relaxed) {
            let mut out = spin_lock(&self.playback.output_recording);
            out.truncate(pos as usize);
            out.resize(pos as usize, 0.0);
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
        self.pipeline.controls.set_contour(pitches);
    }

    /// Restore the default NoteSnapper target.
    pub fn clear_contour(&self) {
        self.pipeline.controls.clear_contour();
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
        a.sync(&self.playback);
        a.input_samples.len() as f64
    }

    /// Analyzed output length in samples. Lags the input by the pipeline
    /// latency during recording; the UI uses it as the repaint watermark
    /// for output tracks.
    pub fn output_len(&self) -> f64 {
        self.analysis.lock().unwrap().output_samples.len() as f64
    }

    /// Detected input pitch per hop (Hz, 0 = unvoiced).
    pub fn input_pitch_track(&self) -> Vec<f32> {
        self.analysis.lock().unwrap().input_pitch.track().to_vec()
    }

    /// Detected output pitch per hop (Hz, 0 = unvoiced).
    pub fn output_pitch_track(&self) -> Vec<f32> {
        self.analysis.lock().unwrap().output_pitch.track().to_vec()
    }

    /// Min/max waveform peaks for a sample range of the input, binned to
    /// `bins` columns; `bins * 2` values interleaved `[min, max]`.
    pub fn input_peaks(&self, start_sample: f64, end_sample: f64, bins: u32) -> Vec<f32> {
        let a = self.analysis.lock().unwrap();
        waveform_peaks(&a.input_samples, start_sample, end_sample, bins as usize)
    }

    /// Same as `input_peaks`, for the produced output.
    pub fn output_peaks(&self, start_sample: f64, end_sample: f64, bins: u32) -> Vec<f32> {
        let a = self.analysis.lock().unwrap();
        waveform_peaks(&a.output_samples, start_sample, end_sample, bins as usize)
    }

    /// Render `width_px` spectrogram columns of the input recording into
    /// `canvas` at `dest_x`, covering the time range starting at
    /// `start_sample` with `samples_per_px` samples per column.
    pub fn draw_input_spectrogram_range(
        &self,
        canvas: &HtmlCanvasElement,
        dest_x: u32,
        width_px: u32,
        start_sample: f64,
        samples_per_px: f64,
    ) -> Result<(), JsValue> {
        self.draw_spec_range(canvas, dest_x, width_px, start_sample, samples_per_px, true)
    }

    /// Same as `draw_input_spectrogram_range`, for the produced output.
    pub fn draw_output_spectrogram_range(
        &self,
        canvas: &HtmlCanvasElement,
        dest_x: u32,
        width_px: u32,
        start_sample: f64,
        samples_per_px: f64,
    ) -> Result<(), JsValue> {
        self.draw_spec_range(canvas, dest_x, width_px, start_sample, samples_per_px, false)
    }

    fn draw_spec_range(
        &self,
        canvas: &HtmlCanvasElement,
        dest_x: u32,
        width_px: u32,
        start_sample: f64,
        samples_per_px: f64,
        input: bool,
    ) -> Result<(), JsValue> {
        if width_px == 0 {
            return Ok(());
        }
        let height = canvas.height();
        let a = &mut *self.analysis.lock().unwrap();
        let samples = if input {
            &a.input_samples
        } else {
            &a.output_samples
        };
        a.spec.render(
            samples,
            start_sample,
            samples_per_px,
            width_px as usize,
            height as usize,
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
