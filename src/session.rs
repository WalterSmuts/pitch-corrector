//! Session analysis: time-indexed data derived from a recording, decoupled
//! from presentation. The UI renders any time range at any zoom from these
//! APIs instead of painting pixels once as audio arrives.
//!
//! Everything here is pure data-in/data-out (no web-sys), so it compiles and
//! is unit-tested on native as well as wasm.

use crate::signal_processing::{YinPitchDetector, BUFFER_SIZE};
use easyfft::dyn_size::realfft::{DynRealDft, DynRealFft};

/// Samples between pitch-analysis hops. Each entry of a pitch track covers
/// one hop; entry `i` is the detection for the window starting at `i * PITCH_HOP`.
pub const PITCH_HOP: usize = BUFFER_SIZE / 2;

/// FFT window for spectrogram rendering. Sized between the DSP buffer (2048,
/// poor low-frequency resolution) and the old display FFT (8192, 170ms of
/// time smear): 4096 gives ~11.7Hz bins at 48kHz with ~85ms windows.
pub const SPEC_WINDOW: usize = 4096;

/// Incremental pitch analysis over a growing recording.
///
/// Feed it the full recording buffer whenever new samples arrive; it only
/// analyzes complete windows it has not seen yet, so calls are cheap.
pub struct PitchTrack {
    detector: YinPitchDetector,
    /// Detected frequency per hop; 0.0 = unvoiced.
    track: Vec<f32>,
    /// Samples fully consumed (start of the next window to analyze).
    consumed: usize,
}

impl PitchTrack {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            detector: YinPitchDetector::with_sample_rate(sample_rate),
            track: Vec::new(),
            consumed: 0,
        }
    }

    /// Analyze any complete, not-yet-seen windows in `samples` (the full
    /// recording so far). Returns the number of new hops produced.
    pub fn analyze(&mut self, samples: &[f32]) -> usize {
        let mut new_hops = 0;
        while self.consumed + BUFFER_SIZE <= samples.len() {
            let window = &samples[self.consumed..self.consumed + BUFFER_SIZE];
            let freq = self.detector.detect(window).unwrap_or(0.0);
            self.track.push(freq);
            self.consumed += PITCH_HOP;
            new_hops += 1;
        }
        new_hops
    }

    /// Detected frequencies, one per hop (0.0 = unvoiced).
    pub fn track(&self) -> &[f32] {
        &self.track
    }

    /// Drop analysis results from `sample_idx` onward so that region is
    /// re-analyzed on the next `analyze` call (used when playback overwrites
    /// the output recording mid-stream).
    pub fn invalidate_from(&mut self, sample_idx: usize) {
        let hop_idx = sample_idx / PITCH_HOP;
        self.track.truncate(hop_idx);
        self.consumed = hop_idx * PITCH_HOP;
    }

    pub fn reset(&mut self) {
        self.track.clear();
        self.consumed = 0;
    }
}

/// Min/max waveform peaks for a sample range, binned to `bins` columns.
/// Returns `bins * 2` values interleaved as `[min, max]` per column, in the
/// style of an audio editor's peak cache. Ranges past the end produce 0,0.
pub fn waveform_peaks(samples: &[f32], start: f64, end: f64, bins: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; bins * 2];
    if bins == 0 || end <= start {
        return out;
    }
    let per_bin = (end - start) / bins as f64;
    for (i, chunk) in out.chunks_exact_mut(2).enumerate() {
        let lo = (start + i as f64 * per_bin).floor().max(0.0) as usize;
        let hi = ((start + (i + 1) as f64 * per_bin).ceil() as usize).min(samples.len());
        if lo >= hi {
            continue;
        }
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for &s in &samples[lo..hi] {
            mn = mn.min(s);
            mx = mx.max(s);
        }
        chunk[0] = mn;
        chunk[1] = mx;
    }
    out
}

/// Classic blue→cyan→green→yellow→red heatmap for spectrogram intensity.
pub fn heatmap(v: u8) -> (u8, u8, u8) {
    match v {
        0..=63 => (0, v * 4, 128 + v * 2),
        64..=127 => (0, 255, 255 - (v - 64) * 4),
        128..=191 => ((v - 128) * 4, 255, 0),
        _ => (255, 255 - (v - 192) * 4, 0),
    }
}

/// Renders spectrogram pixels for an arbitrary time range of a recording.
/// Holds pre-allocated FFT scratch so repeated (incremental) renders don't
/// allocate.
pub struct SpectrogramRenderer {
    scratch: Vec<f32>,
    spectrum: DynRealDft<f32>,
    hann: Vec<f32>,
}

impl Default for SpectrogramRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectrogramRenderer {
    pub fn new() -> Self {
        let scratch = vec![0.0f32; SPEC_WINDOW];
        let spectrum = scratch.real_fft();
        let hann = (0..SPEC_WINDOW)
            .map(|i| 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / SPEC_WINDOW as f32).cos()))
            .collect();
        Self {
            scratch,
            spectrum,
            hann,
        }
    }

    /// Render `width` columns × `height` rows into `rgba` (row-major RGBA,
    /// resized to `width * height * 4`). Column `x` shows the FFT of a
    /// window centered at `start_sample + (x + 0.5) * samples_per_px`.
    /// The y axis is log-frequency, highest at the top; `f_lo..f_hi`
    /// (fractions of the full log-frequency axis, 0 = lowest bin,
    /// 1 = Nyquist) select the vertical window, so the UI can zoom the
    /// frequency axis. Out-of-range samples are treated as silence.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        samples: &[f32],
        start_sample: f64,
        samples_per_px: f64,
        width: usize,
        height: usize,
        f_lo: f32,
        f_hi: f32,
        rgba: &mut Vec<u8>,
    ) {
        rgba.clear();
        rgba.resize(width * height * 4, 0);
        if width == 0 || height == 0 {
            return;
        }

        let num_bins = self.spectrum.get_frequency_bins().len();
        let log_min = 1.0f32.ln();
        let log_max = (num_bins as f32).ln();

        for x in 0..width {
            let center = start_sample + (x as f64 + 0.5) * samples_per_px;
            let win_start = center as i64 - (SPEC_WINDOW / 2) as i64;
            for (i, (dst, w)) in self.scratch.iter_mut().zip(&self.hann).enumerate() {
                let idx = win_start + i as i64;
                let s = if idx >= 0 && (idx as usize) < samples.len() {
                    samples[idx as usize]
                } else {
                    0.0
                };
                *dst = s * w;
            }
            self.scratch.real_fft_using(&mut self.spectrum);
            let bins = self.spectrum.get_frequency_bins();

            for y in 0..height {
                let t = 1.0 - (y as f32 / height as f32);
                let tt = f_lo + t * (f_hi - f_lo);
                let bin_f = (log_min + tt * (log_max - log_min)).exp();
                let bin_lo = (bin_f as usize).min(num_bins - 2);
                let frac = bin_f - bin_lo as f32;
                let mag_lo = bins[bin_lo].norm();
                let mag_hi = bins[bin_lo + 1].norm();
                let mut mag = (mag_lo * (1.0 - frac) + mag_hi * frac) / SPEC_WINDOW as f32;
                mag *= bin_f.sqrt();

                let power = mag * mag;
                let db = if power > 1e-20 {
                    10.0 * power.log10()
                } else {
                    -100.0
                };
                let intensity = ((db + 100.0) * (255.0 / 80.0)).clamp(0.0, 255.0) as u8;
                let (r, g, b) = heatmap(intensity);
                let o = (y * width + x) * 4;
                rgba[o] = r;
                rgba[o + 1] = g;
                rgba[o + 2] = b;
                rgba[o + 3] = 255;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sample_rate: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn pitch_track_detects_sine_and_is_incremental() {
        let sr = 48_000.0;
        let samples = sine(220.0, sr, 48_000);

        // One-shot analysis.
        let mut all = PitchTrack::new(sr);
        all.analyze(&samples);
        let expected_hops = (samples.len() - BUFFER_SIZE) / PITCH_HOP + 1;
        assert_eq!(all.track().len(), expected_hops);
        // Skip the first few hops (detector warm-up/gate) then expect 220Hz.
        for (i, &f) in all.track().iter().enumerate().skip(3) {
            assert!(
                (f - 220.0).abs() < 3.0,
                "hop {i}: expected ~220Hz, got {f}"
            );
        }

        // Incremental analysis over growing buffers must agree exactly.
        let mut inc = PitchTrack::new(sr);
        for end in (0..=samples.len()).step_by(1000) {
            inc.analyze(&samples[..end]);
        }
        inc.analyze(&samples);
        assert_eq!(inc.track(), all.track());
    }

    #[test]
    fn pitch_track_reports_silence_as_unvoiced() {
        let sr = 48_000.0;
        // Voiced then silence: the silent tail must come back 0.0.
        let mut samples = sine(220.0, sr, 24_000);
        samples.extend(std::iter::repeat(0.0).take(24_000));
        let mut pt = PitchTrack::new(sr);
        pt.analyze(&samples);
        let track = pt.track();
        let tail = &track[track.len() - 5..];
        assert!(
            tail.iter().all(|&f| f == 0.0),
            "silent tail should be unvoiced, got {tail:?}"
        );
    }

    #[test]
    fn pitch_track_invalidate_from_reanalyzes() {
        let sr = 48_000.0;
        let samples = sine(220.0, sr, 48_000);
        let mut pt = PitchTrack::new(sr);
        pt.analyze(&samples);
        let before = pt.track().to_vec();
        pt.invalidate_from(24_000);
        assert!(pt.track().len() < before.len());
        pt.analyze(&samples);
        assert_eq!(pt.track().len(), before.len());
    }

    #[test]
    fn waveform_peaks_min_max() {
        // Alternating +1/-1: every bin must see min=-1, max=+1.
        let samples: Vec<f32> = (0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let peaks = waveform_peaks(&samples, 0.0, 1000.0, 10);
        assert_eq!(peaks.len(), 20);
        for pair in peaks.chunks_exact(2) {
            assert_eq!(pair[0], -1.0);
            assert_eq!(pair[1], 1.0);
        }
        // Past-the-end range renders as silence, not garbage.
        let empty = waveform_peaks(&samples, 2000.0, 3000.0, 4);
        assert!(empty.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn spectrogram_renders_tone_hotter_than_silence() {
        let sr = 48_000.0;
        let samples = sine(440.0, sr, 48_000);
        let mut r = SpectrogramRenderer::new();
        let mut rgba = Vec::new();
        // Two columns: one centered in the tone, one far past the end (silence).
        r.render(&samples, 12_000.0, 1.0, 1, 64, 0.0, 1.0, &mut rgba);
        let tone_energy: u32 = rgba.iter().step_by(4).map(|&v| v as u32).sum();
        r.render(&samples, 500_000.0, 1.0, 1, 64, 0.0, 1.0, &mut rgba);
        let silence_energy: u32 = rgba.iter().step_by(4).map(|&v| v as u32).sum();
        assert!(
            tone_energy > silence_energy,
            "tone column ({tone_energy}) should be hotter than silence ({silence_energy})"
        );
        assert_eq!(rgba.len(), 64 * 4);
    }
}
