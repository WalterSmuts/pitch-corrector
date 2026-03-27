use crate::pitch_correction::{Notes, PitchContour, PitchCorrector, PitchTarget};
use crate::signal_processing::{compose, DisplayProcessor, StreamProcessor, YinPitchDetector};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use easyfft::dyn_size::realfft::{DynRealDft, DynRealFft};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

const SPECTROGRAM_SIZE: usize = 8192;
const CONTOUR_SIZE: usize = 2048;

// Pre-computed RGB strings for heatmap: avoids format!() per pixel
static HEATMAP_LUT: std::sync::LazyLock<[String; 256]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|i| {
        let (r, g, b) = heatmap(i as u8);
        format!("rgb({r},{g},{b})")
    })
});

static GRID_ALPHA_STRONG: &str = "rgba(255,255,255,0.3)";
static GRID_ALPHA_WEAK: &str = "rgba(255,255,255,0.1)";

/// Pre-allocated scratch buffers for zero-alloc drawing
struct DrawState {
    spec_scratch: Vec<f32>,
    spec_spectrum: DynRealDft<f32>,
    contour_scratch: Vec<f32>,
    output_detector: YinPitchDetector,
    input_detector: YinPitchDetector,
}

#[wasm_bindgen]
pub struct WebPitchCorrector {
    input_stream: cpal::Stream,
    output_stream: cpal::Stream,
    spectrogram_buffer: Arc<Mutex<[f32; SPECTROGRAM_SIZE]>>,
    contour_buffer: Arc<Mutex<[f32; CONTOUR_SIZE]>>,
    input_spectrogram_buffer: Arc<Mutex<[f32; SPECTROGRAM_SIZE]>>,
    input_contour_buffer: Arc<Mutex<[f32; CONTOUR_SIZE]>>,
    shift_control: Arc<AtomicU32>,
    target_log: Arc<Mutex<Vec<f32>>>,
    notes_control: Arc<AtomicU16>,
    target_handle: Arc<Mutex<Arc<dyn PitchTarget>>>,
    default_target: Arc<dyn PitchTarget>,
    sweep_active: Arc<AtomicBool>,
    input_active: Arc<AtomicBool>,
    recording: Arc<Mutex<Vec<f32>>>,
    playback_pos: Arc<AtomicU32>,
    playing: Arc<AtomicBool>,
    draw_state: Mutex<DrawState>,
}

#[wasm_bindgen]
impl WebPitchCorrector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebPitchCorrector, JsValue> {
        console_log::init_with_level(log::Level::Info).ok();

        let spectrogram_display: DisplayProcessor<SPECTROGRAM_SIZE> = DisplayProcessor::new();
        let spectrogram_buffer = spectrogram_display.clone_display_buffer();

        let contour_display: DisplayProcessor<CONTOUR_SIZE> = DisplayProcessor::new();
        let contour_buffer = contour_display.clone_display_buffer();

        let input_contour_display: DisplayProcessor<CONTOUR_SIZE> = DisplayProcessor::new();
        let input_contour_buffer = input_contour_display.clone_display_buffer();

        let input_spectrogram_display: DisplayProcessor<SPECTROGRAM_SIZE> = DisplayProcessor::new();
        let input_spectrogram_buffer = input_spectrogram_display.clone_display_buffer();

        let corrector = PitchCorrector::new();
        let shift_control = corrector.shift_control();
        let target_log = corrector.target_log();
        let notes_control = corrector.as_note_snapper().unwrap().notes_control();
        let target_handle = corrector.target_handle();
        let default_target = target_handle.lock().unwrap().clone();

        // Pipeline: input_contour -> input_spectrogram -> corrector -> contour -> spectrogram
        let processor = Arc::new(compose(
            input_contour_display,
            compose(
                input_spectrogram_display,
                compose(corrector, compose(contour_display, spectrogram_display)),
            ),
        ));
        let input_processor = processor.clone();
        let output_processor = processor.clone();

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

        let sweep_active = Arc::new(AtomicBool::new(false));
        let sweep_flag = sweep_active.clone();
        // Pack two f32s: [sample_counter, accumulated_phase]
        let sweep_counter = Arc::new(AtomicU32::new(0.0f32.to_bits()));
        let sweep_phase = Arc::new(AtomicU32::new(0.0f32.to_bits()));
        let sweep_counter_clone = sweep_counter.clone();
        let sweep_phase_clone = sweep_phase.clone();

        let input_active = Arc::new(AtomicBool::new(true));
        let input_active_flag = input_active.clone();

        let recording: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let recording_clone = recording.clone();

        let input_stream = input_device
            .build_input_stream(
                input_config.into(),
                move |data: &[f32], _| {
                    if !input_active_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    if sweep_flag.load(Ordering::Relaxed) {
                        let mut counter =
                            f32::from_bits(sweep_counter_clone.load(Ordering::Relaxed));
                        let mut phase = f32::from_bits(sweep_phase_clone.load(Ordering::Relaxed));
                        for _ in data {
                            let freq = 200.0 - (counter / 480000.0) * 150.0;
                            phase += freq / 48000.0;
                            phase -= phase.floor();
                            let sample = (phase * std::f32::consts::TAU).sin() * 0.5;
                            input_processor.push_sample(sample);
                            if let Ok(mut rec) = recording_clone.try_lock() {
                                rec.push(sample);
                            }
                            counter += 1.0;
                            if counter >= 480000.0 {
                                counter = 0.0;
                            }
                        }
                        sweep_counter_clone.store(counter.to_bits(), Ordering::Relaxed);
                        sweep_phase_clone.store(phase.to_bits(), Ordering::Relaxed);
                    } else {
                        if let Ok(mut rec) = recording_clone.try_lock() {
                            rec.extend_from_slice(data);
                        }
                        for &sample in data {
                            input_processor.push_sample(sample);
                        }
                    }
                },
                |err| log::error!("Input error: {}", err),
                None,
            )
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        let playback_rec = recording.clone();
        let playback_pos_out = Arc::new(AtomicU32::new(0));
        let playback_pos_clone = playback_pos_out.clone();
        let playback_playing = Arc::new(AtomicBool::new(false));
        let playback_flag = playback_playing.clone();

        let output_stream = output_device
            .build_output_stream(
                output_config.into(),
                move |data: &mut [f32], _| {
                    if playback_flag.load(Ordering::Relaxed) {
                        if let Ok(rec) = playback_rec.try_lock() {
                            let mut p = playback_pos_clone.load(Ordering::Relaxed) as usize;
                            for _ in 0..data.len() {
                                if p < rec.len() {
                                    output_processor.push_sample(rec[p]);
                                    p += 1;
                                }
                            }
                            if p >= rec.len() {
                                playback_flag.store(false, Ordering::Relaxed);
                            }
                            playback_pos_clone.store(p as u32, Ordering::Relaxed);
                        }
                    }
                    for sample in data.iter_mut() {
                        match output_processor.pop_sample() {
                            Some(s) => *sample = s,
                            None => {
                                log::warn!("Output callback: underrun — inserting silence");
                                *sample = 0.0;
                            }
                        }
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
            spectrogram_buffer,
            contour_buffer,
            input_spectrogram_buffer,
            input_contour_buffer,
            shift_control,
            target_log,
            notes_control,
            target_handle,
            default_target,
            sweep_active,
            input_active,
            recording,
            playback_pos: playback_pos_out,
            playing: playback_playing,
            draw_state: Mutex::new({
                let spec_scratch = vec![0.0f32; SPECTROGRAM_SIZE];
                let spec_spectrum = spec_scratch.real_fft();
                DrawState {
                    spec_scratch,
                    spec_spectrum,
                    contour_scratch: vec![0.0f32; CONTOUR_SIZE],
                    output_detector: YinPitchDetector::new(),
                    input_detector: YinPitchDetector::new(),
                }
            }),
        })
    }

    pub fn set_shift(&self, semitones: f32) {
        self.shift_control
            .store(semitones.to_bits(), Ordering::Relaxed);
    }

    pub fn get_shift(&self) -> f32 {
        f32::from_bits(self.shift_control.load(Ordering::Relaxed))
    }

    /// Returns the recorded target contour (one entry per phase vocoder hop)
    /// and clears the log.
    pub fn take_target_contour(&self) -> Vec<f32> {
        std::mem::take(&mut *self.target_log.lock().unwrap())
    }

    pub fn clear_target_log(&self) {
        self.target_log.lock().unwrap().clear();
    }

    pub fn set_notes(&self, bits: u16) {
        self.notes_control.store(bits, Ordering::Relaxed);
    }

    pub fn get_notes(&self) -> u16 {
        self.notes_control.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.input_active.store(false, Ordering::Relaxed);
        let _ = self.input_stream.pause();
        let _ = self.output_stream.pause();
    }

    pub fn recording_len(&self) -> usize {
        self.recording.lock().unwrap().len()
    }

    pub fn play_recording(&self) -> Result<(), JsValue> {
        if self.recording.lock().unwrap().is_empty() {
            return Ok(());
        }
        self.input_active.store(false, Ordering::Relaxed);
        self.playing.store(true, Ordering::Relaxed);
        let _ = self.output_stream.play();
        Ok(())
    }

    pub fn stop_playback(&self) {
        self.playing.store(false, Ordering::Relaxed);
        let _ = self.output_stream.pause();
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    pub fn playback_progress(&self) -> f32 {
        let len = self.recording_len();
        if len == 0 {
            return 0.0;
        }
        self.playback_pos.load(Ordering::Relaxed) as f32 / len as f32
    }

    pub fn seek(&self, fraction: f32) {
        let len = self.recording_len() as f32;
        let pos = (fraction.clamp(0.0, 1.0) * len) as u32;
        self.playback_pos.store(pos, Ordering::Relaxed);
    }

    pub fn set_sweep(&self, active: bool) {
        self.sweep_active.store(active, Ordering::Relaxed);
    }

    pub fn scale_bits(preset: &str, root: u8) -> u16 {
        let root_note = Notes::BY_INDEX[root as usize % 12];
        match preset {
            "off" => Notes::empty().bits(),
            "chromatic" => Notes::chromatic().bits(),
            "major" => Notes::major(root_note).bits(),
            "minor" => Notes::minor(root_note).bits(),
            "pentatonic" => Notes::pentatonic(root_note).bits(),
            _ => Notes::chromatic().bits(),
        }
    }

    pub fn snap_to_scale(freq: f32, note_bits: u16) -> f32 {
        use crate::pitch_correction::nearest_note;
        let notes = Notes::from_bits_truncate(note_bits);
        nearest_note(freq, notes)
    }

    /// Set a pitch contour as the active target for playback.
    /// `contour` is a JS Float32Array of target frequencies (one per hop).
    pub fn set_contour(&self, contour: &[f32]) {
        let pc = Arc::new(PitchContour::new(contour.to_vec()));
        *self.target_handle.lock().unwrap() = pc;
    }

    /// Restore the default NoteSnapper target.
    pub fn clear_contour(&self) {
        *self.target_handle.lock().unwrap() = self.default_target.clone();
    }

    pub fn draw_spectrogram(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        let mut ds = self.draw_state.lock().unwrap();
        draw_spectrogram_from(canvas, column_x, &self.spectrogram_buffer, &mut ds);
    }

    pub fn draw_input_spectrogram(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        let mut ds = self.draw_state.lock().unwrap();
        draw_spectrogram_from(canvas, column_x, &self.input_spectrogram_buffer, &mut ds);
    }
    pub fn draw_pitch_contour(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        let ds = &mut *self.draw_state.lock().unwrap();
        draw_contour(
            canvas,
            column_x,
            &self.contour_buffer,
            "rgb(50,255,120)",
            &mut ds.output_detector,
            &mut ds.contour_scratch,
        );
    }

    pub fn draw_input_contour(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        let ds = &mut *self.draw_state.lock().unwrap();
        draw_contour(
            canvas,
            column_x,
            &self.input_contour_buffer,
            "rgb(255,150,50)",
            &mut ds.input_detector,
            &mut ds.contour_scratch,
        );
    }

    pub fn draw_waveform(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        draw_waveform_from(
            canvas,
            column_x,
            &self.spectrogram_buffer,
            "rgb(50,255,120)",
        );
    }

    pub fn draw_input_waveform(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        draw_waveform_from(
            canvas,
            column_x,
            &self.input_spectrogram_buffer,
            "rgb(255,150,50)",
        );
    }
}

fn draw_spectrogram_from(
    canvas: &HtmlCanvasElement,
    column_x: f32,
    buffer: &Arc<Mutex<[f32; SPECTROGRAM_SIZE]>>,
    ds: &mut DrawState,
) {
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();

    let height = canvas.height() as usize;

    // Item 2: copy into pre-allocated scratch instead of .to_vec()
    {
        let buf = buffer.lock().unwrap();
        ds.spec_scratch.copy_from_slice(&*buf);
    }

    let len = ds.spec_scratch.len() as f32;
    for (i, sample) in ds.spec_scratch.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / len).cos());
        *sample *= w;
    }

    // Item 3: in-place FFT using pre-allocated spectrum
    ds.spec_scratch.real_fft_using(&mut ds.spec_spectrum);
    let bins = ds.spec_spectrum.get_frequency_bins();
    let num_bins = bins.len();

    let min_bin = 1.0f32;
    let max_bin = num_bins as f32;
    let log_min = min_bin.ln();
    let log_max = max_bin.ln();

    for y_pixel in 0..height {
        let t = 1.0 - (y_pixel as f32 / height as f32);
        let log_bin = log_min + t * (log_max - log_min);
        let bin_f = log_bin.exp();
        let bin_lo = (bin_f as usize).min(num_bins - 2);
        let bin_hi = bin_lo + 1;
        let frac = bin_f - bin_lo as f32;

        let mag_lo = bins[bin_lo].norm();
        let mag_hi = bins[bin_hi].norm();
        let mut mag = (mag_lo * (1.0 - frac) + mag_hi * frac) / SPECTROGRAM_SIZE as f32;
        mag *= bin_f.sqrt();

        let power = mag * mag;
        let db = if power > 1e-20 {
            10.0 * power.log10()
        } else {
            -100.0
        };
        let intensity = ((db + 100.0) * (255.0 / 80.0)).clamp(0.0, 255.0) as u8;

        // Item 1: pre-computed RGB string lookup instead of format!()
        ctx.set_fill_style_str(&HEATMAP_LUT[intensity as usize]);
        ctx.fill_rect(column_x as f64, y_pixel as f64, 1.0, 1.0);
    }

    // Write-head bar
    let bar_x = ((column_x as u32 + 1) % canvas.width()) as f64;
    ctx.set_fill_style_str("rgba(255,255,255,0.8)");
    ctx.fill_rect(bar_x, 0.0, 4.0, height as f64);
}

fn draw_contour(
    canvas: &HtmlCanvasElement,
    column_x: f32,
    buffer: &Arc<Mutex<[f32; CONTOUR_SIZE]>>,
    color: &str,
    detector: &mut YinPitchDetector,
    scratch: &mut Vec<f32>,
) {
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();

    let height = canvas.height() as f32;

    // Item 2: copy into pre-allocated scratch instead of .to_vec()
    {
        let buf = buffer.lock().unwrap();
        scratch.resize(buf.len(), 0.0);
        scratch.copy_from_slice(&*buf);
    }

    // Item 4: reuse detector instead of YinPitchDetector::new()
    let pitch = detector.detect(scratch);

    // Background
    ctx.set_fill_style_str("rgb(10,10,20)");
    ctx.fill_rect(column_x as f64, 0.0, 2.0, height as f64);

    // Semitone grid lines with note names
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    for semitone in 0..12 {
        let y = height * (1.0 - semitone as f32 / 12.0);
        // Item 5: pre-computed alpha strings
        let alpha_str = if semitone == 0 || semitone == 4 || semitone == 7 {
            GRID_ALPHA_STRONG
        } else {
            GRID_ALPHA_WEAK
        };
        ctx.set_stroke_style_str(alpha_str);
        ctx.begin_path();
        ctx.move_to(column_x as f64, y as f64);
        ctx.line_to((column_x + 2.0) as f64, y as f64);
        ctx.stroke();

        if column_x < 2.0 {
            ctx.set_fill_style_str("rgba(255,255,255,0.5)");
            ctx.set_font("10px monospace");
            let _ = ctx.fill_text(note_names[semitone], 4.0, (y - 3.0) as f64);
        }
    }

    // Detected pitch wrapped to one octave
    if let Some(freq) = pitch {
        let semitones_from_c = 12.0 * (freq / 16.3516).log2();
        let semitone_in_octave = semitones_from_c.rem_euclid(12.0);
        let y = height * (1.0 - semitone_in_octave / 12.0);
        ctx.set_fill_style_str(color);
        ctx.begin_path();
        let _ = ctx.arc(
            (column_x + 1.0) as f64,
            y as f64,
            4.0,
            0.0,
            std::f64::consts::TAU,
        );
        ctx.fill();
    }

    // Write-head bar
    let bar_x = ((column_x as u32 + 2) % canvas.width()) as f64;
    ctx.set_fill_style_str("rgba(255,255,255,0.8)");
    ctx.fill_rect(bar_x, 0.0, 4.0, height as f64);
}

fn draw_waveform_from(
    canvas: &HtmlCanvasElement,
    column_x: f32,
    buffer: &Arc<Mutex<[f32; SPECTROGRAM_SIZE]>>,
    color: &str,
) {
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();

    let height = canvas.height() as f32;
    let mid = height / 2.0;
    let buf = buffer.lock().unwrap();

    // Find peak amplitude for this frame
    let peak = buf
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
        .max(0.001);

    // RMS for filled bar, peak for outline
    let rms = (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt();
    let rms_h = (rms / peak) * mid;
    let peak_h = mid;

    // Background
    ctx.set_fill_style_str("rgb(10,10,20)");
    ctx.fill_rect(column_x as f64, 0.0, 1.0, height as f64);

    // Peak bar (dim)
    ctx.set_global_alpha(0.3);
    ctx.set_fill_style_str(color);
    ctx.fill_rect(
        column_x as f64,
        (mid - peak_h) as f64,
        1.0,
        (peak_h * 2.0) as f64,
    );

    // RMS bar (bright)
    ctx.set_global_alpha(1.0);
    ctx.fill_rect(
        column_x as f64,
        (mid - rms_h) as f64,
        1.0,
        (rms_h * 2.0) as f64,
    );

    // Write-head bar
    let bar_x = ((column_x as u32 + 1) % canvas.width()) as f64;
    ctx.set_fill_style_str("rgba(255,255,255,0.8)");
    ctx.fill_rect(bar_x, 0.0, 4.0, height as f64);
}

fn heatmap(v: u8) -> (u8, u8, u8) {
    match v {
        0..=63 => (0, v * 4, 128 + v * 2),
        64..=127 => (0, 255, 255 - (v - 64) * 4),
        128..=191 => ((v - 128) * 4, 255, 0),
        _ => (255, 255 - (v - 192) * 4, 0),
    }
}
