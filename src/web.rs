use crate::signal_processing::{DisplayProcessor, StreamProcessor};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

const DISPLAY_SIZE: usize = 8192;

#[wasm_bindgen]
pub struct WebPitchCorrector {
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
    display_buffer: Arc<Mutex<[f32; DISPLAY_SIZE]>>,
    write_index: Arc<AtomicUsize>,
}

#[wasm_bindgen]
impl WebPitchCorrector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebPitchCorrector, JsValue> {
        console_log::init_with_level(log::Level::Info).ok();

        let display: DisplayProcessor<DISPLAY_SIZE> = DisplayProcessor::new();
        let display_buffer = display.clone_display_buffer();
        let write_index = display.clone_write_index();
        let input_processor = Arc::new(display);
        let output_processor = input_processor.clone();

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

        // Separate buffer to capture raw input for spectrogram
        let input_display_buffer = Arc::new(Mutex::new([0.0f32; DISPLAY_SIZE]));
        let input_write_index = Arc::new(AtomicUsize::new(0));
        let input_buf_clone = input_display_buffer.clone();
        let input_idx_clone = input_write_index.clone();

        let input_stream = input_device
            .build_input_stream(
                input_config.into(),
                move |data: &[f32], _| {
                    // Write directly to spectrogram buffer
                    {
                        let mut buf = input_buf_clone.lock().unwrap();
                        for &sample in data {
                            let idx =
                                input_idx_clone.fetch_add(1, Ordering::Relaxed) % DISPLAY_SIZE;
                            buf[idx] = sample;
                        }
                    }
                    for &sample in data {
                        input_processor.push_sample(sample);
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
            display_buffer: input_display_buffer,
            write_index: input_write_index,
        })
    }

    pub fn draw_spectrogram(&self, canvas: &HtmlCanvasElement, column_x: f32) {
        use easyfft::dyn_size::DynFft;

        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into()
            .unwrap();

        let height = canvas.height() as usize;
        let raw_buffer = self.display_buffer.lock().unwrap();
        let idx = self.write_index.load(Ordering::Relaxed) % DISPLAY_SIZE;

        let mut buffer = vec![0.0f32; DISPLAY_SIZE];
        buffer[..DISPLAY_SIZE - idx].copy_from_slice(&raw_buffer[idx..]);
        buffer[DISPLAY_SIZE - idx..].copy_from_slice(&raw_buffer[..idx]);
        drop(raw_buffer);

        let len = buffer.len() as f32;
        for (i, sample) in buffer.iter_mut().enumerate() {
            let w = 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / len).cos());
            *sample *= w;
        }

        let spectrum = buffer.fft();
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
            let mut mag = (mag_lo * (1.0 - frac) + mag_hi * frac) / DISPLAY_SIZE as f32;
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
}

fn heatmap(v: u8) -> (u8, u8, u8) {
    match v {
        0..=63 => (0, v * 4, 128 + v * 2),
        64..=127 => (0, 255, 255 - (v - 64) * 4),
        128..=191 => ((v - 128) * 4, 255, 0),
        _ => (255, 255 - (v - 192) * 4, 0),
    }
}
