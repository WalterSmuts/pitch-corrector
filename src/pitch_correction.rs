use crate::music::{Interval, Note, Pitch, Scale};
use crate::signal_processing::{
    PhaseVocoderPitchShifter, StreamProcessor, YinPitchDetector, BUFFER_SIZE,
};
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
        let prev_frame = Mutex::new(vec![0.0f32; BUFFER_SIZE]);
        let snapper = Mutex::new(NoteSnapper::new());
        let ratio_fn: RatioFn = Box::new(move |frame: &[f32]| {
            let shift_ratio = controls_clone.shift.lock().unwrap().to_ratio();

            // Try detection on current frame first; fall back to doubled buffer
            let detected = {
                let mut det = detector.lock().unwrap();
                let short = det.detect(frame);
                let mut prev = prev_frame.lock().unwrap();
                let result = short.or_else(|| {
                    let mut extended = Vec::with_capacity(prev.len() + frame.len());
                    extended.extend_from_slice(&prev);
                    extended.extend_from_slice(frame);
                    det.detect(&extended)
                });
                prev.copy_from_slice(frame);
                result
            };

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
    fn pitch_corrector_off_is_transparent_for_sweep() {
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

        assert!(
            (best_sim - 1.0).abs() < 0.05,
            "Corrector with empty notes should be transparent for sweep \
             (best similarity {best_sim:.3} at offset {best_off})"
        );
    }

    #[test]
    fn pitch_corrector_snaps_descending_sweep_to_scale() {
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
        assert!(
            accuracy > 0.75,
            "Expected >75% of detected pitches on pentatonic C scale, \
             but only {correct}/{checked} ({:.0}%) were within 0.5 semitones",
            accuracy * 100.0
        );
    }
}
