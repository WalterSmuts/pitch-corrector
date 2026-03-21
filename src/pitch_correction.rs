use crate::signal_processing::{PhaseVocoderPitchShifter, YinPitchDetector};

/// Scale notes: only fifths (C, G)
const SCALE_OFFSETS: [f32; 2] = [0.0, 7.0];

/// Find the nearest scale note frequency
fn nearest_scale_note(freq: f32) -> f32 {
    let semitones_from_c0 = 12.0 * (freq / 16.3516).log2();
    let octave = (semitones_from_c0 / 12.0).floor();
    let semitone_in_octave = semitones_from_c0 - octave * 12.0;

    let mut best_offset = SCALE_OFFSETS[0];
    let mut best_dist = f32::MAX;
    for &offset in &SCALE_OFFSETS {
        let dist = (semitone_in_octave - offset).abs();
        if dist < best_dist {
            best_dist = dist;
            best_offset = offset;
        }
        let wrap_dist = (semitone_in_octave - (offset + 12.0)).abs();
        if wrap_dist < best_dist {
            best_dist = wrap_dist;
            best_offset = offset + 12.0;
        }
    }

    let target_semitones = octave * 12.0 + best_offset;
    16.3516 * (2.0f32).powf(target_semitones / 12.0)
}

fn pitch_correction_ratio(frame: &[f32]) -> f32 {
    let detector = YinPitchDetector::new();
    match detector.detect(frame) {
        Some(freq) => {
            let target = nearest_scale_note(freq);
            let ratio = target / freq;
            log::info!(
                "Pitch: {:.1}Hz -> {:.1}Hz (ratio: {:.3})",
                freq,
                target,
                ratio
            );
            ratio
        }
        None => 1.0,
    }
}

pub fn new_pitch_corrector() -> PhaseVocoderPitchShifter<impl Fn(&[f32]) -> f32 + Send + Sync> {
    PhaseVocoderPitchShifter::with_ratio_fn(pitch_correction_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_processing::StreamProcessor;
    use std::f32::consts::TAU;

    const BUFFER_SIZE: usize = 1024;
    const SAMPLE_RATE: usize = 44100;

    #[test]
    fn nearest_scale_note_finds_c() {
        approx::assert_abs_diff_eq!(nearest_scale_note(261.63), 261.63, epsilon = 1.0);
    }

    #[test]
    fn nearest_scale_note_snaps_to_c_or_g() {
        // F4 = 349.23 Hz should snap to C4 or G4
        let corrected = nearest_scale_note(349.23);
        assert!(
            (corrected - 261.63).abs() < 2.0 || (corrected - 392.0).abs() < 2.0,
            "Expected C4 or G4, got {corrected}"
        );
    }

    #[test]
    fn pitch_corrector_produces_output() {
        let processor = new_pitch_corrector();

        let num_samples = BUFFER_SIZE * 10;
        for i in 0..num_samples {
            let sample = (TAU * 445.0 * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
        }

        let mut output = Vec::new();
        while let Some(s) = processor.pop_sample() {
            output.push(s);
        }

        assert!(
            output.len() > BUFFER_SIZE,
            "Expected output, got {} samples",
            output.len()
        );

        let max_amp: f32 = output.iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(max_amp > 0.01, "Output is silent: max amp {max_amp}");
    }
}
