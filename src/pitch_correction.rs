use crate::music::{Interval, Note, Pitch, Scale};
use crate::signal_processing::{PhaseVocoderPitchShifter, StreamProcessor, YinPitchDetector};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// Strategy for choosing a target pitch given the detected pitch.
trait PitchTarget: Send + Sync {
    /// Given the detected frequency, return the target pitch.
    /// Return `None` to leave pitch unchanged.
    fn target(&self, detected_freq: f32) -> Option<Pitch>;
}

/// Snaps detected pitch to the nearest note in a scale, with hysteresis.
struct NoteSnapper {
    scale: Arc<Mutex<Scale>>,
    prev_target: Mutex<Option<Pitch>>,
}

impl NoteSnapper {
    fn new(scale: Arc<Mutex<Scale>>) -> Self {
        Self {
            scale,
            prev_target: Mutex::new(None),
        }
    }
}

impl PitchTarget for NoteSnapper {
    fn target(&self, detected_freq: f32) -> Option<Pitch> {
        let current = *self.scale.lock().unwrap();
        if current.is_empty() {
            return None;
        }

        let target = current.nearest_pitch(detected_freq);

        // Schmitt trigger: only switch note if detected pitch has
        // crossed more than half a semitone past the previous target
        let mut prev = self.prev_target.lock().unwrap();
        let result = if let Some(p) = *prev {
            if target != p {
                let dist = (12.0 * (detected_freq / p.to_freq()).log2()).abs();
                if dist < 0.5 {
                    p
                } else {
                    *prev = Some(target);
                    target
                }
            } else {
                target
            }
        } else {
            *prev = Some(target);
            target
        };

        Some(result)
    }
}

/// Follows a pre-defined sequence of target frequencies indexed by hop.
struct PitchContour {
    /// Target pitch per hop (`None` = passthrough).
    contour: Vec<Option<Pitch>>,
    /// Counts ratio_fn invocations (one per phase vocoder hop).
    hop_count: AtomicU32,
}

impl PitchContour {
    fn new(contour: Vec<Option<Pitch>>) -> Self {
        Self {
            contour,
            hop_count: AtomicU32::new(0),
        }
    }
}

impl PitchTarget for PitchContour {
    fn target(&self, _detected_freq: f32) -> Option<Pitch> {
        if self.contour.is_empty() {
            return None;
        }
        let hop = self.hop_count.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = hop.min(self.contour.len() - 1);
        self.contour[idx]
    }
}

type RatioFn = Box<dyn Fn(&[f32]) -> f32 + Send + Sync>;

/// Remote control for a `PitchCorrector` that has been moved into a pipeline.
pub struct PitchCorrectorControls {
    shift: Mutex<Interval>,
    scale: Arc<Mutex<Scale>>,
    target_pitch_contour: Mutex<Vec<Option<Pitch>>>,
    target: Mutex<Arc<dyn PitchTarget>>,
    default_target: Arc<dyn PitchTarget>,
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
        *self.target.lock().unwrap() = Arc::new(PitchContour::new(contour));
    }

    pub fn clear_contour(&self) {
        *self.target.lock().unwrap() = self.default_target.clone();
    }

    pub fn take_target_pitch_contour(&self) -> Vec<Option<Pitch>> {
        std::mem::take(&mut *self.target_pitch_contour.lock().unwrap())
    }

    pub fn clear_target_pitch_contour(&self) {
        self.target_pitch_contour.lock().unwrap().clear();
    }

    pub fn snap_to_scale(&self, freq: f32) -> f32 {
        self.get_scale().nearest_pitch(freq).to_freq()
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
        let scale = Arc::new(Mutex::new(scale));
        let snapper: Arc<dyn PitchTarget> = Arc::new(NoteSnapper::new(scale.clone()));
        let controls = Arc::new(PitchCorrectorControls {
            shift: Mutex::new(Interval::UNISON),
            scale,
            target_pitch_contour: Mutex::new(Vec::new()),
            target: Mutex::new(snapper.clone()),
            default_target: snapper,
        });

        let controls_clone = controls.clone();
        let detector = Mutex::new(YinPitchDetector::new());
        let ratio_fn: RatioFn = Box::new(move |frame: &[f32]| {
            let shift_ratio = controls_clone.shift.lock().unwrap().to_ratio();
            let current_target = controls_clone.target.lock().unwrap().clone();

            let correction = match detector.lock().unwrap().detect(frame) {
                Some(freq) => match current_target.target(freq) {
                    Some(pitch) => {
                        if let Ok(mut log) = controls_clone.target_pitch_contour.try_lock() {
                            log.push(Some(pitch));
                        }
                        pitch.to_freq() / freq
                    }
                    None => {
                        if let Ok(mut log) = controls_clone.target_pitch_contour.try_lock() {
                            log.push(None);
                        }
                        1.0
                    }
                },
                None => {
                    if let Ok(mut log) = controls_clone.target_pitch_contour.try_lock() {
                        log.push(None);
                    }
                    1.0
                }
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
    use std::f32::consts::TAU;

    const BUFFER_SIZE: usize = 1024;
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
