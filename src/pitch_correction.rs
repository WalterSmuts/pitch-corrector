use crate::music::{Interval, Note, Pitch, Scale};
use crate::signal_processing::{PhaseVocoderPitchShifter, StreamProcessor, YinPitchDetector};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// Snaps detected pitch to the nearest note in a scale, with hysteresis.
struct NoteSnapper {
    prev_target: Option<Pitch>,
}

impl NoteSnapper {
    fn new() -> Self {
        Self { prev_target: None }
    }

    /// Snap `detected_freq` to the nearest note in `scale`.
    /// Returns `None` if the scale is empty (passthrough).
    fn snap(&mut self, detected_freq: f32, scale: Scale) -> Option<Pitch> {
        if scale.is_empty() {
            return None;
        }

        let target = scale.nearest_pitch(detected_freq);

        // Schmitt trigger: only switch note if detected pitch has
        // crossed more than half a semitone past the previous target
        let result = if let Some(p) = self.prev_target {
            if target != p {
                let dist = (12.0 * (detected_freq / p.to_freq()).log2()).abs();
                if dist < 0.5 {
                    p
                } else {
                    self.prev_target = Some(target);
                    target
                }
            } else {
                target
            }
        } else {
            self.prev_target = Some(target);
            target
        };

        Some(result)
    }
}

type RatioFn = Box<dyn Fn(&[f32]) -> f32 + Send + Sync>;

/// Remote control for a `PitchCorrector` that has been moved into a pipeline.
pub struct PitchCorrectorControls {
    shift: Mutex<Interval>,
    scale: Mutex<Scale>,
    target_pitch_contour: Mutex<Vec<Option<Pitch>>>,
    contour: Mutex<Vec<Option<Pitch>>>,
    contour_hop: AtomicUsize,
}

impl PitchCorrectorControls {
    pub fn set_shift(&self, interval: Interval) {
        *self.shift.lock().unwrap() = interval;
    }

    pub fn get_shift(&self) -> Interval {
        *self.shift.lock().unwrap()
    }

    pub fn set_scale(&self, scale: Scale) {
        *self.scale.lock().unwrap() = scale;
    }

    pub fn get_scale(&self) -> Scale {
        *self.scale.lock().unwrap()
    }

    pub fn set_contour(&self, contour: Vec<Option<Pitch>>) {
        *self.contour.lock().unwrap() = contour;
        self.contour_hop.store(0, Ordering::Relaxed);
    }

    pub fn clear_contour(&self) {
        self.contour.lock().unwrap().clear();
        self.contour_hop.store(0, Ordering::Relaxed);
    }

    pub fn take_target_pitch_contour(&self) -> Vec<Option<Pitch>> {
        std::mem::take(&mut *self.target_pitch_contour.lock().unwrap())
    }

    pub fn clear_target_pitch_contour(&self) {
        self.target_pitch_contour.lock().unwrap().clear();
    }
}

pub struct PitchCorrector {
    processor: PhaseVocoderPitchShifter<RatioFn>,
    controls: Arc<PitchCorrectorControls>,
}

impl Default for PitchCorrector {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchCorrector {
    pub fn new() -> Self {
        Self::with_scale(Scale::pentatonic(Note::C))
    }

    pub fn with_scale(scale: Scale) -> Self {
        let controls = Arc::new(PitchCorrectorControls {
            shift: Mutex::new(Interval::UNISON),
            scale: Mutex::new(scale),
            target_pitch_contour: Mutex::new(Vec::new()),
            contour: Mutex::new(Vec::new()),
            contour_hop: AtomicUsize::new(0),
        });

        let controls_clone = controls.clone();
        let detector = Mutex::new(YinPitchDetector::new());
        let snapper = Mutex::new(NoteSnapper::new());
        let ratio_fn: RatioFn = Box::new(move |frame: &[f32]| {
            let shift_ratio = controls_clone.shift.lock().unwrap().to_ratio();
            let detected = detector.lock().unwrap().detect(frame);

            // Check for active contour, otherwise snap to scale
            let target_pitch = {
                let contour = controls_clone.contour.lock().unwrap();
                if !contour.is_empty() {
                    let hop = controls_clone.contour_hop.fetch_add(1, Ordering::Relaxed);
                    contour[hop.min(contour.len() - 1)]
                } else {
                    let scale = *controls_clone.scale.lock().unwrap();
                    detected.and_then(|freq| snapper.lock().unwrap().snap(freq, scale))
                }
            };

            controls_clone
                .target_pitch_contour
                .lock()
                .unwrap()
                .push(target_pitch);

            let correction = match (target_pitch, detected) {
                (Some(pitch), Some(freq)) => pitch.to_freq() / freq,
                _ => 1.0,
            };
            correction * shift_ratio
        });
        let processor = PhaseVocoderPitchShifter::with_ratio_fn(ratio_fn);
        PitchCorrector {
            processor,
            controls,
        }
    }

    /// Extract the controls handle before moving this into a pipeline.
    pub fn controls(&self) -> Arc<PitchCorrectorControls> {
        self.controls.clone()
    }
}

impl StreamProcessor for PitchCorrector {
    fn push_sample(&self, sample: f32) {
        self.processor.push_sample(sample);
    }

    fn pop_sample(&self) -> Option<f32> {
        self.processor.pop_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_processing::BUFFER_SIZE;
    use std::f32::consts::TAU;

    const SAMPLE_RATE: usize = 44100;

    // --- Perf thresholds ---
    const PERF_CORRECTOR_TRANSPARENCY: f32 = 0.01; // max |similarity - 1.0|
    const PERF_SNAPPING_ACCURACY: f32 = 0.90; // min fraction on scale
    const PERF_TRACKING_PASS: f32 = 0.80; // min accuracy per rate
    const PERF_MIN_TRACKING_RATE: f32 = 2.0; // Hz
    const PERF_NOISE_PASS: f32 = 0.75; // min accuracy per noise level
    const PERF_MIN_NOISE_TOLERANCE: f32 = 0.3; // amplitude

    #[test]
    fn set_scale_at_runtime() {
        let corrector = PitchCorrector::new();
        let controls = corrector.controls();
        controls.set_scale(Scale::major(Note::C));
        assert_eq!(controls.get_scale(), Scale::major(Note::C));
    }

    #[test]
    fn pitch_corrector_produces_output() {
        let corrector = PitchCorrector::new();

        let num_samples = BUFFER_SIZE * 10;
        for i in 0..num_samples {
            let sample = (TAU * 445.0 * i as f32 / SAMPLE_RATE as f32).sin();
            corrector.push_sample(sample);
        }

        let mut output = Vec::new();
        while let Some(s) = corrector.pop_sample() {
            output.push(s);
        }

        assert!(output.len() > BUFFER_SIZE);
        let max_amp: f32 = output.iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(max_amp > 0.01);
    }

    #[test]
    fn perf_pitch_corrector_off_is_transparent_for_sweep() {
        let corrector = PitchCorrector::with_scale(Scale::empty());

        let num_samples = BUFFER_SIZE * 40;
        let mut phase = 0.0f32;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let freq = 100.0 + (i as f32 / num_samples as f32) * 900.0;
                phase += freq / SAMPLE_RATE as f32;
                phase -= phase.floor();
                (phase * TAU).sin() * 0.5
            })
            .collect();

        let mut output = Vec::new();
        for &s in &input {
            corrector.push_sample(s);
            while let Some(o) = corrector.pop_sample() {
                output.push(o);
            }
        }

        let skip = BUFFER_SIZE * 4;
        let compare_len = BUFFER_SIZE * 5;
        assert!(
            output.len() > skip + compare_len,
            "Not enough output: {}",
            output.len()
        );
        let delay = input.len() - output.len();
        // Find best alignment by searching for peak cross-correlation
        let compare_len = BUFFER_SIZE * 5;
        let mid = output.len() / 2;
        let mut best_sim = f32::MIN;
        let mut best_off = 0usize;
        for off in delay.saturating_sub(BUFFER_SIZE)..delay + BUFFER_SIZE {
            if mid + compare_len > output.len() || mid + off + compare_len > input.len() {
                continue;
            }
            let out_slice = &output[mid..mid + compare_len];
            let in_slice = &input[mid + off..mid + off + compare_len];
            let cross: f32 = in_slice.iter().zip(out_slice).map(|(a, b)| a * b).sum();
            let auto: f32 = in_slice.iter().map(|a| a * a).sum();
            let sim = cross / auto;
            if sim > best_sim {
                best_sim = sim;
                best_off = off;
            }
        }

        eprintln!(
            "[PERF] corrector_transparency: similarity={best_sim:.4} (threshold: >{:.2})",
            1.0 - PERF_CORRECTOR_TRANSPARENCY
        );
        assert!(
            (best_sim - 1.0).abs() < PERF_CORRECTOR_TRANSPARENCY,
            "Corrector with empty notes should be transparent for sweep \
             (best similarity {best_sim:.3} at offset {best_off})"
        );
    }

    #[test]
    fn perf_pitch_corrector_snaps_descending_sweep_to_scale() {
        use crate::signal_processing::YinPitchDetector;

        let corrector = PitchCorrector::with_scale(Scale::pentatonic(Note::C));

        // Descending sweep 200Hz -> 50Hz
        let num_samples = BUFFER_SIZE * 80;
        let mut phase = 0.0f32;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let freq = 200.0 - (i as f32 / num_samples as f32) * 150.0;
                phase += freq / SAMPLE_RATE as f32;
                phase -= phase.floor();
                (phase * TAU).sin() * 0.5
            })
            .collect();

        let mut output = Vec::new();
        for &s in &input {
            corrector.push_sample(s);
            while let Some(o) = corrector.pop_sample() {
                output.push(o);
            }
        }

        // Detect pitch at several points in the output
        let mut detector = YinPitchDetector::new();
        let pentatonic_c = Scale::pentatonic(Note::C);
        let mut checked = 0;
        let mut correct = 0;

        let skip = BUFFER_SIZE * 8;
        let step = BUFFER_SIZE * 4;
        let mut pos = skip;
        while pos + 2048 <= output.len() {
            if let Some(freq) = detector.detect(&output[pos..pos + 2048]) {
                let target = pentatonic_c.nearest_pitch(freq).to_freq();
                let semitone_error = (12.0 * (freq / target).log2()).abs();
                checked += 1;
                if semitone_error < 0.5 {
                    correct += 1;
                }
            }
            pos += step;
        }

        assert!(checked > 5, "Not enough pitch detections: {checked}");
        let accuracy = correct as f32 / checked as f32;
        eprintln!(
            "[PERF] corrector_snapping_accuracy: {correct}/{checked} ({:.1}%) (threshold: >{:.0}%)",
            accuracy * 100.0,
            PERF_SNAPPING_ACCURACY * 100.0
        );
        assert!(
            accuracy > PERF_SNAPPING_ACCURACY,
            "Expected >{:.0}% of detected pitches on pentatonic C scale, \
             but only {correct}/{checked} ({:.0}%) were within 0.5 semitones",
            PERF_SNAPPING_ACCURACY * 100.0,
            accuracy * 100.0
        );
    }

    /// Measures how quickly the pitch corrector adapts to changing targets.
    ///
    /// Generates a sine whose pitch swings between two pentatonic notes with
    /// increasing vibrato rate. At each rate we measure what fraction of the
    /// output is correctly snapped. Reports the fastest vibrato rate (Hz)
    /// that still achieves ≥80% accuracy.
    #[test]
    fn perf_pitch_corrector_tracking_bandwidth() {
        use crate::signal_processing::YinPitchDetector;

        let corrector = PitchCorrector::with_scale(Scale::pentatonic(Note::C));

        // Swing between G3 (196Hz) and A3 (220Hz) — adjacent pentatonic notes
        let center = (196.0f32.ln() + 220.0f32.ln()) / 2.0;
        let swing = (220.0f32.ln() - 196.0f32.ln()) / 2.0;

        // Test vibrato rates from 0.5Hz to 16Hz in doublings
        let rates: Vec<f32> = (0..6).map(|i| 0.5 * 2.0f32.powi(i)).collect();
        let samples_per_rate = BUFFER_SIZE * 40;
        let total_samples = samples_per_rate * rates.len();

        // Generate input
        let mut audio_phase = 0.0f32;
        let mut input = Vec::with_capacity(total_samples);
        for (ri, &rate) in rates.iter().enumerate() {
            for j in 0..samples_per_rate {
                let t = (ri * samples_per_rate + j) as f32 / SAMPLE_RATE as f32;
                let vibrato = (TAU * rate * t).sin();
                let freq = (center + swing * vibrato).exp();
                audio_phase += freq / SAMPLE_RATE as f32;
                audio_phase -= audio_phase.floor();
                input.push((audio_phase * TAU).sin() * 0.5);
            }
        }

        // Process
        let mut output = Vec::with_capacity(total_samples);
        for &s in &input {
            corrector.push_sample(s);
            while let Some(o) = corrector.pop_sample() {
                output.push(o);
            }
        }

        let delay = input.len() - output.len();
        let pentatonic_c = Scale::pentatonic(Note::C);
        let mut detector = YinPitchDetector::new();
        let mut best_rate = 0.0f32;

        for (ri, &rate) in rates.iter().enumerate() {
            let region_start = ri * samples_per_rate;
            // Skip first quarter of each region for settling
            let analysis_start = (region_start + samples_per_rate / 4).saturating_sub(delay);
            let analysis_end = ((ri + 1) * samples_per_rate).saturating_sub(delay);

            let step = BUFFER_SIZE;
            let mut checked = 0;
            let mut correct = 0;
            let mut pos = analysis_start;
            while pos + BUFFER_SIZE <= analysis_end.min(output.len()) {
                if let Some(freq) = detector.detect(&output[pos..pos + BUFFER_SIZE]) {
                    let target = pentatonic_c.nearest_pitch(freq).to_freq();
                    let semitone_error = (12.0 * (freq / target).log2()).abs();
                    checked += 1;
                    if semitone_error < 0.5 {
                        correct += 1;
                    }
                }
                pos += step;
            }

            let accuracy = if checked > 0 {
                correct as f32 / checked as f32
            } else {
                0.0
            };
            eprintln!(
                "[PERF] corrector_tracking_{rate:.0}hz: {correct}/{checked} ({:5.1}%)",
                accuracy * 100.0
            );
            if accuracy >= PERF_TRACKING_PASS {
                best_rate = rate;
            }
        }

        eprintln!("[PERF] corrector_tracking_bandwidth: {best_rate:.1}Hz (threshold: >={PERF_MIN_TRACKING_RATE:.1}Hz)");

        assert!(
            best_rate >= PERF_MIN_TRACKING_RATE,
            "Pitch corrector should track at least {PERF_MIN_TRACKING_RATE:.1}Hz vibrato, but only managed {best_rate:.1}Hz"
        );
    }

    /// Measures how much additive noise the pitch corrector can tolerate
    /// while tracking vibrato at PERF_MIN_TRACKING_RATE.
    ///
    /// Generates a sine swinging between G3 and A3 with increasing noise.
    /// Reports the highest noise amplitude where the corrector still
    /// achieves ≥PERF_NOISE_PASS accuracy.
    #[test]
    fn perf_pitch_corrector_noise_tolerance() {
        use crate::signal_processing::YinPitchDetector;
        use rand::Rng;

        let pentatonic_c = Scale::pentatonic(Note::C);

        // Same vibrato as tracking test, fixed at PERF_MIN_TRACKING_RATE
        let center = (196.0f32.ln() + 220.0f32.ln()) / 2.0;
        let swing = (220.0f32.ln() - 196.0f32.ln()) / 2.0;

        // Noise levels as fraction of signal amplitude: 0.0, 0.1, ..., 1.0
        let levels: Vec<f32> = (0..=10).map(|i| i as f32 * 0.1).collect();
        let samples_per_level = BUFFER_SIZE * 40;

        let mut best_noise = 0.0f32;
        let mut rng = rand::rng();

        for &noise_amp in &levels {
            let corrector = PitchCorrector::with_scale(pentatonic_c);

            let mut phase = 0.0f32;
            let mut input = Vec::with_capacity(samples_per_level);
            for i in 0..samples_per_level {
                let t = i as f32 / SAMPLE_RATE as f32;
                let vibrato = (TAU * PERF_MIN_TRACKING_RATE * t).sin();
                let freq = (center + swing * vibrato).exp();
                phase += freq / SAMPLE_RATE as f32;
                phase -= phase.floor();
                let signal = (phase * TAU).sin() * 0.5;
                let noise = (rng.random::<f32>() * 2.0 - 1.0) * noise_amp * 0.5;
                input.push(signal + noise);
            }

            let mut output = Vec::new();
            for &s in &input {
                corrector.push_sample(s);
                while let Some(o) = corrector.pop_sample() {
                    output.push(o);
                }
            }

            let delay = input.len() - output.len();
            let skip = (samples_per_level / 4).saturating_sub(delay);
            let mut detector = YinPitchDetector::new();
            let mut checked = 0;
            let mut correct = 0;
            let mut pos = skip;
            while pos + BUFFER_SIZE <= output.len() {
                if let Some(f) = detector.detect(&output[pos..pos + BUFFER_SIZE]) {
                    let target = pentatonic_c.nearest_pitch(f).to_freq();
                    let err = (12.0 * (f / target).log2()).abs();
                    checked += 1;
                    if err < 0.5 {
                        correct += 1;
                    }
                }
                pos += BUFFER_SIZE;
            }

            let accuracy = if checked > 0 {
                correct as f32 / checked as f32
            } else {
                0.0
            };
            let snr_db = if noise_amp > 0.0 {
                20.0 * (1.0 / noise_amp).log10()
            } else {
                f32::INFINITY
            };
            eprintln!(
                "[PERF] corrector_noise_{noise_amp:.1}: {correct}/{checked} ({:5.1}%) SNR={snr_db:.1}dB",
                accuracy * 100.0
            );
            if accuracy >= PERF_NOISE_PASS {
                best_noise = noise_amp;
            }
        }

        eprintln!("[PERF] corrector_noise_tolerance: {best_noise:.1} (threshold: >={PERF_MIN_NOISE_TOLERANCE})");

        assert!(
            best_noise >= PERF_MIN_NOISE_TOLERANCE,
            "Pitch corrector should tolerate at least {PERF_MIN_NOISE_TOLERANCE} noise amplitude, \
             but only managed {best_noise:.1}"
        );
    }
}
