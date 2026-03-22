use crate::pitch_correction::{Notes, PitchCorrector};
use crate::signal_processing::{compose, DisplayProcessor, StreamProcessor, YinPitchDetector};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

const SPECTROGRAM_SIZE: usize = 8192;
const CONTOUR_SIZE: usize = 2048;

#[wasm_bindgen]
pub struct WebPitchCorrector {
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
    spectrogram_buffer: Arc<Mutex<[f32; SPECTROGRAM_SIZE]>>,
    spectrogram_index: Arc<AtomicUsize>,
    contour_buffer: Arc<Mutex<[f32; CONTOUR_SIZE]>>,
    contour_index: Arc<AtomicUsize>,
    input_spectrogram_buffer: Arc<Mutex<[f32; SPECTROGRAM_SIZE]>>,
    input_spectrogram_index: Arc<AtomicUsize>,
    input_contour_buffer: Arc<Mutex<[f32; CONTOUR_SIZE]>>,
    input_contour_index: Arc<AtomicUsize>,
    shift_control: Arc<AtomicU32>,
    notes_control: Arc<AtomicU16>,
    sweep_active: Arc<AtomicBool>,
}

#[wasm_bindgen]
impl WebPitchCorrector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebPitchCorrector, JsValue> {
        console_log::init_with_level(log::Level::Info).ok();

        let spectrogram_display: DisplayProcessor<SPECTROGRAM_SIZE> = DisplayProcessor::new();
        let spectrogram_buffer = spectrogram_display.clone_display_buffer();
        let spectrogram_index = spectrogram_display.clone_write_index();

        let contour_display: DisplayProcessor<CONTOUR_SIZE> = DisplayProcessor::new();
        let contour_buffer = contour_display.clone_display_buffer();
        let contour_index = contour_display.clone_write_index();

        let input_contour_display: DisplayProcessor<CONTOUR_SIZE> = DisplayProcessor::new();
        let input_contour_buffer = input_contour_display.clone_display_buffer();
        let input_contour_index = input_contour_display.clone_write_index();

        let input_spectrogram_display: DisplayProcessor<SPECTROGRAM_SIZE> = DisplayProcessor::new();
        let input_spectrogram_buffer = input_spectrogram_display.clone_display_buffer();
        let input_spectrogram_index = input_spectrogram_display.clone_write_index();

        let corrector = PitchCorrector::new();
        let shift_control = corrector.shift_control();
        let notes_control = corrector.notes_control();

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
        let sweep_phase = Arc::new(AtomicU32::new(0.0f32.to_bits()));
        let sweep_phase_clone = sweep_phase.clone();

        let input_stream = input_device
            .build_input_stream(
                input_config.into(),
                move |data: &[f32], _| {
                    if sweep_flag.load(Ordering::Relaxed) {
                        // Generate rising sine sweep: 100Hz to 1000Hz over ~10 seconds
                        let mut phase = f32::from_bits(sweep_phase_clone.load(Ordering::Relaxed));
                        for _ in data {
                            let freq = 100.0 + (phase / 480000.0) * 900.0; // 10s at 48kHz
                            let sample =
                                (phase * freq * std::f32::consts::TAU / 48000.0).sin() * 0.5;
                            input_processor.push_sample(sample);
                            phase += 1.0;
                            if phase >= 480000.0 {
                                phase = 0.0;
                            }
                        }
                        sweep_phase_clone.store(phase.to_bits(), Ordering::Relaxed);
                    } else {
                        for &sample in data {
                            input_processor.push_sample(sample);
                        }
                    }
                },
                |err| log::error!("Input error: {}", err),
                None,
            )
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        let output_stream = output_device
            .build_output_stream(
                output_config.into(),
                move |data: &mut [f32], _| {
                    for sample in data.iter_mut() {
                        *sample = output_processor.pop_sample().unwrap_or(0.0);
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
            _input_stream: input_stream,
            _output_stream: output_stream,
            spectrogram_buffer,
            spectrogram_index,
            contour_buffer,
            contour_index,
            input_spectrogram_buffer,
            input_spectrogram_index,
            input_contour_buffer,
            input_contour_index,
            shift_control,
            notes_control,
            sweep_active,
        })
    }

    pub fn set_shift(&self, semitones: f32) {
        self.shift_control
            .store(semitones.to_bits(), Ordering::Relaxed);
    }

    pub fn get_shift(&self) -> f32 {
        f32::from_bits(self.shift_control.load(Ordering::Relaxed))
    }

    pub fn set_notes(&self, bits: u16) {
        self.notes_control.store(bits, Ordering::Relaxed);
    }

    pub fn get_notes(&self) -> u16 {
        self.notes_control.load(Ordering::Relaxed)
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

    pub fn draw_spectrogram(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        draw_spectrogram_from(
            canvas,
            column_x,
            &self.spectrogram_buffer,
            &self.spectrogram_index,
        );
    }

    pub fn draw_input_spectrogram(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        draw_spectrogram_from(
            canvas,
            column_x,
            &self.input_spectrogram_buffer,
            &self.input_spectrogram_index,
        );
    }
    pub fn draw_pitch_contour(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        draw_contour(
            canvas,
            column_x,
            &self.contour_buffer,
            &self.contour_index,
            "rgb(50,255,120)",
        );
    }

    pub fn draw_input_contour(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        draw_contour(
            canvas,
            column_x,
            &self.input_contour_buffer,
            &self.input_contour_index,
            "rgb(255,150,50)",
        );
    }
}

fn draw_spectrogram_from(
    canvas: &HtmlCanvasElement,
    column_x: f32,
    buffer: &Arc<Mutex<[f32; SPECTROGRAM_SIZE]>>,
    index: &Arc<AtomicUsize>,
) {
    use easyfft::dyn_size::DynFft;

    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();

    let height = canvas.height() as usize;
    let raw_buffer = buffer.lock().unwrap();
    let idx = index.load(Ordering::Relaxed) % SPECTROGRAM_SIZE;

    let mut buf = vec![0.0f32; SPECTROGRAM_SIZE];
    buf[..SPECTROGRAM_SIZE - idx].copy_from_slice(&raw_buffer[idx..]);
    buf[SPECTROGRAM_SIZE - idx..].copy_from_slice(&raw_buffer[..idx]);
    drop(raw_buffer);

    let len = buf.len() as f32;
    for (i, sample) in buf.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / len).cos());
        *sample *= w;
    }

    let spectrum = buf.fft();
    let num_bins = spectrum.len() / 2;

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

        let mag_lo = spectrum[bin_lo].norm();
        let mag_hi = spectrum[bin_hi].norm();
        let mut mag = (mag_lo * (1.0 - frac) + mag_hi * frac) / SPECTROGRAM_SIZE as f32;
        mag *= bin_f.sqrt();

        let power = mag * mag;
        let db = if power > 1e-20 {
            10.0 * power.log10()
        } else {
            -100.0
        };
        let intensity = ((db + 100.0) * (255.0 / 80.0)).clamp(0.0, 255.0) as u8;

        let (r, g, b) = heatmap(intensity);
        ctx.set_fill_style_str(&format!("rgb({r},{g},{b})"));
        ctx.fill_rect(column_x as f64, y_pixel as f64, 1.0, 1.0);
    }
}

fn draw_contour(
    canvas: &HtmlCanvasElement,
    column_x: f32,
    buffer: &Arc<Mutex<[f32; CONTOUR_SIZE]>>,
    index: &Arc<AtomicUsize>,
    color: &str,
) {
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap();

    let height = canvas.height() as f32;
    let raw_buffer = buffer.lock().unwrap();
    let idx = index.load(Ordering::Relaxed) % CONTOUR_SIZE;
    let mut buf = vec![0.0f32; CONTOUR_SIZE];
    buf[..CONTOUR_SIZE - idx].copy_from_slice(&raw_buffer[idx..]);
    buf[CONTOUR_SIZE - idx..].copy_from_slice(&raw_buffer[..idx]);
    drop(raw_buffer);

    let energy: f32 = buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32;

    let mut detector = YinPitchDetector::new();
    let pitch = if energy > 0.0001 {
        detector.detect(&buf)
    } else {
        None
    };

    // Background
    ctx.set_fill_style_str("rgb(10,10,20)");
    ctx.fill_rect(column_x as f64, 0.0, 2.0, height as f64);

    // Semitone grid lines with note names
    let note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    for semitone in 0..12 {
        let y = height * (1.0 - semitone as f32 / 12.0);
        let alpha = if semitone == 0 || semitone == 4 || semitone == 7 {
            0.3
        } else {
            0.1
        };
        ctx.set_stroke_style_str(&format!("rgba(255,255,255,{alpha})"));
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
}

fn heatmap(v: u8) -> (u8, u8, u8) {
    match v {
        0..=63 => (0, v * 4, 128 + v * 2),
        64..=127 => (0, 255, 255 - (v - 64) * 4),
        128..=191 => ((v - 128) * 4, 255, 0),
        _ => (255, 255 - (v - 192) * 4, 0),
    }
}
