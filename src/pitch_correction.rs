use crate::signal_processing::{PhaseVocoderPitchShifter, StreamProcessor, YinPitchDetector};
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Notes: u16 {
        const C  = 1 << 0;
        const CS = 1 << 1;
        const D  = 1 << 2;
        const DS = 1 << 3;
        const E  = 1 << 4;
        const F  = 1 << 5;
        const FS = 1 << 6;
        const G  = 1 << 7;
        const GS = 1 << 8;
        const A  = 1 << 9;
        const AS = 1 << 10;
        const B  = 1 << 11;
    }
}

impl Notes {
    pub const BY_INDEX: [Notes; 12] = [
        Notes::C,
        Notes::CS,
        Notes::D,
        Notes::DS,
        Notes::E,
        Notes::F,
        Notes::FS,
        Notes::G,
        Notes::GS,
        Notes::A,
        Notes::AS,
        Notes::B,
    ];

    /// Build a note set from intervals relative to a root note.
    pub fn from_intervals(intervals: &[u8], root: Notes) -> Self {
        let root_idx = root.bits().trailing_zeros() as u8;
        let mut result = Notes::empty();
        for &interval in intervals {
            result |= Notes::BY_INDEX[((root_idx + interval) % 12) as usize];
        }
        result
    }

    pub fn chromatic() -> Self {
        Notes::all()
    }

    pub fn major(root: Notes) -> Self {
        Self::from_intervals(&[0, 2, 4, 5, 7, 9, 11], root)
    }

    pub fn minor(root: Notes) -> Self {
        Self::from_intervals(&[0, 2, 3, 5, 7, 8, 10], root)
    }

    pub fn pentatonic(root: Notes) -> Self {
        Self::from_intervals(&[0, 2, 4, 7, 9], root)
    }
}

fn nearest_note(freq: f32, notes: Notes) -> f32 {
    if notes.is_empty() {
        return freq;
    }
    let semitones_from_c0 = 12.0 * (freq / 16.3516).log2();
    let octave = (semitones_from_c0 / 12.0).floor();
    let semitone_in_octave = semitones_from_c0 - octave * 12.0;

    let mut best_offset = 0.0f32;
    let mut best_dist = f32::MAX;
    for i in 0..12u8 {
        if !notes.contains(Notes::BY_INDEX[i as usize]) {
            continue;
        }
        let note_f = i as f32;
        for &candidate in &[note_f, note_f + 12.0, note_f - 12.0] {
            let dist = (semitone_in_octave - candidate).abs();
            if dist < best_dist {
                best_dist = dist;
                best_offset = candidate;
            }
        }
    }

    let target_semitones = octave * 12.0 + best_offset;
    16.3516 * (2.0f32).powf(target_semitones / 12.0)
}

type RatioFn = Box<dyn Fn(&[f32]) -> f32 + Send + Sync>;

pub struct PitchCorrector {
    shift_semitones: Arc<AtomicU32>,
    note_set: Arc<AtomicU16>,
    processor: PhaseVocoderPitchShifter<RatioFn>,
}

impl PitchCorrector {
    pub fn new() -> Self {
        Self::with_notes(Notes::pentatonic(Notes::C))
    }

    pub fn with_notes(notes: Notes) -> Self {
        let shift = Arc::new(AtomicU32::new(0.0f32.to_bits()));
        let note_set = Arc::new(AtomicU16::new(notes.bits()));
        let shift_clone = shift.clone();
        let notes_clone = note_set.clone();
        let prev_target = AtomicU32::new(0.0f32.to_bits());
        let detector = Mutex::new(YinPitchDetector::new());
        let ratio_fn: RatioFn = Box::new(move |frame: &[f32]| {
            let current_notes = Notes::from_bits_truncate(notes_clone.load(Ordering::Relaxed));
            let semitones = f32::from_bits(shift_clone.load(Ordering::Relaxed));
            let shift_ratio = (2.0f32).powf(semitones / 12.0);

            if current_notes.is_empty() {
                return shift_ratio;
            }

            let correction = match detector.lock().unwrap().detect(frame) {
                Some(freq) => {
                    let target = nearest_note(freq, current_notes);

                    // Schmitt trigger: only switch note if detected pitch has
                    // crossed more than half a semitone past the previous target
                    let prev = f32::from_bits(prev_target.load(Ordering::Relaxed));
                    let final_target = if prev > 0.0 && target != prev {
                        let dist_from_prev = (12.0 * (freq / prev).log2()).abs();
                        if dist_from_prev < 0.5 {
                            prev
                        } else {
                            prev_target.store(target.to_bits(), Ordering::Relaxed);
                            target
                        }
                    } else {
                        prev_target.store(target.to_bits(), Ordering::Relaxed);
                        target
                    };

                    final_target / freq
                }
                None => 1.0,
            };
            correction * shift_ratio
        });
        let processor = PhaseVocoderPitchShifter::with_ratio_fn(ratio_fn);
        PitchCorrector {
            shift_semitones: shift,
            note_set,
            processor,
        }
    }

    #[allow(dead_code)]
    pub fn set_semitones(&self, semitones: f32) {
        self.shift_semitones
            .store(semitones.to_bits(), Ordering::Relaxed);
    }

    pub fn shift_control(&self) -> Arc<AtomicU32> {
        self.shift_semitones.clone()
    }

    #[allow(dead_code)]
    pub fn set_notes(&self, notes: Notes) {
        self.note_set.store(notes.bits(), Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub fn notes_control(&self) -> Arc<AtomicU16> {
        self.note_set.clone()
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
    fn notes_chromatic_has_all() {
        assert_eq!(Notes::chromatic().bits().count_ones(), 12);
    }

    #[test]
    fn notes_major_c() {
        let ns = Notes::major(Notes::C);
        assert!(ns.contains(Notes::C));
        assert!(!ns.contains(Notes::CS));
        assert!(ns.contains(Notes::D));
        assert!(!ns.contains(Notes::DS));
        assert!(ns.contains(Notes::E));
        assert!(ns.contains(Notes::F));
        assert!(!ns.contains(Notes::FS));
        assert!(ns.contains(Notes::G));
        assert!(!ns.contains(Notes::GS));
        assert!(ns.contains(Notes::A));
        assert!(!ns.contains(Notes::AS));
        assert!(ns.contains(Notes::B));
    }

    #[test]
    fn notes_major_g() {
        let ns = Notes::major(Notes::G);
        assert!(ns.contains(Notes::G));
        assert!(ns.contains(Notes::A));
        assert!(ns.contains(Notes::B));
        assert!(ns.contains(Notes::C));
        assert!(ns.contains(Notes::D));
        assert!(ns.contains(Notes::E));
        assert!(ns.contains(Notes::FS));
        assert!(!ns.contains(Notes::F));
    }

    #[test]
    fn notes_off_returns_input() {
        assert_eq!(nearest_note(445.0, Notes::empty()), 445.0);
    }

    #[test]
    fn nearest_note_chromatic_snaps_to_a() {
        let corrected = nearest_note(445.0, Notes::chromatic());
        approx::assert_abs_diff_eq!(corrected, 440.0, epsilon = 2.0);
    }

    #[test]
    fn nearest_note_pentatonic_c() {
        let corrected = nearest_note(349.23, Notes::pentatonic(Notes::C));
        assert!(
            (corrected - 329.63).abs() < 2.0 || (corrected - 392.0).abs() < 2.0,
            "Expected E4 or G4, got {corrected}"
        );
    }

    #[test]
    fn custom_note_set() {
        // Just C and G
        let ns = Notes::C | Notes::G;
        let corrected = nearest_note(349.23, ns);
        assert!(
            (corrected - 261.63).abs() < 2.0 || (corrected - 392.0).abs() < 2.0,
            "Expected C4 or G4, got {corrected}"
        );
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
    fn set_notes_at_runtime() {
        let corrector = PitchCorrector::new();
        corrector.set_notes(Notes::major(Notes::C));
        assert_eq!(
            Notes::from_bits_truncate(corrector.note_set.load(Ordering::Relaxed)),
            Notes::major(Notes::C)
        );
    }

    #[test]
    fn pitch_corrector_off_is_transparent_for_sweep() {
        let corrector = PitchCorrector::with_notes(Notes::empty());

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

        let corrector = PitchCorrector::with_notes(Notes::pentatonic(Notes::C));

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
        let pentatonic_c = Notes::pentatonic(Notes::C);
        let mut checked = 0;
        let mut correct = 0;

        let skip = BUFFER_SIZE * 8;
        let step = BUFFER_SIZE * 4;
        let mut pos = skip;
        while pos + 2048 <= output.len() {
            if let Some(freq) = detector.detect(&output[pos..pos + 2048]) {
                let target = nearest_note(freq, pentatonic_c);
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
