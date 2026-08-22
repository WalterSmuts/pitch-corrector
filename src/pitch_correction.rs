use crate::music::{Interval, Note, Pitch, Scale, SimpleInterval};
use crate::signal_processing::{
    PhaseVocoderPitchShifter, StreamProcessor, YinPitchDetector, HOP_SIZE,
};
use crossbeam_queue::ArrayQueue;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// Sentinel for "no target this hop" in `hop_target_semis`.
const NO_TARGET: i32 = i32::MIN;

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
    /// Pitch shift as total semitones, read lock-free on the audio thread.
    shift_semitones: AtomicI32,
    /// Scale note-set bitmask (12 bits), read lock-free on the audio thread.
    scale_bits: AtomicU32,
    /// Per-hop target-pitch log, written from the real-time audio thread.
    /// A lock-free bounded queue keeps that write allocation- and lock-free;
    /// it is drained from the UI thread on stop/clear. Entries are dropped if
    /// the log fills (bounded memory) rather than growing unboundedly.
    target_pitch_contour: ArrayQueue<Option<Pitch>>,
    /// Per-hop publication for the harmony voices, which run as parallel
    /// pipelines and must agree with the main voice's analysis: the detected
    /// frequency (f32 bits; 0 = unvoiced) and the final post-shift target
    /// as whole semitones from C0 (NO_TARGET = none). The main ratio_fn
    /// writes these each hop before the harmony ratio_fns read them.
    hop_detected_bits: AtomicU32,
    hop_target_semis: AtomicI32,
    /// Enabled harmony voices: bit0 = 3rd, bit1 = 5th, bit2 = octave.
    harmony_mask: AtomicU32,
    /// Per-hop produced-pitch log of every voice — [main, 3rd, 5th, octave]
    /// (Hz; 0 = silent/disabled) — written lock- and alloc-free from the
    /// audio thread and drained by the UI. The output pitch view plots
    /// these directly: the DSP knows what each voice produces, and running
    /// a monophonic detector on the mixed output would garble.
    voice_pitch_log: [ArrayQueue<f32>; 4],
    /// Diatonic (walk the scale) vs absolute (fixed semitones) harmony.
    harmony_in_key: AtomicBool,
    /// Edited target contour (post-correction playback). Written wholesale by
    /// the UI and read by index on the audio thread. It is only populated
    /// during post-correction playback and swapped rarely, so the lock is
    /// low-contention; making a variable-length buffer fully lock-free would
    /// need an arc-swap-style atomic pointer swap (extra dependency).
    contour: Mutex<Vec<Option<Pitch>>>,
    contour_hop: AtomicUsize,
}

impl PitchCorrectorControls {
    pub fn set_shift(&self, interval: Interval) {
        self.shift_semitones
            .store(interval.semitones(), Ordering::Relaxed);
    }

    pub fn get_shift(&self) -> Interval {
        let s = self.shift_semitones.load(Ordering::Relaxed);
        Interval::compound(
            SimpleInterval::ALL[s.rem_euclid(12) as usize],
            s.div_euclid(12) as i8,
        )
    }

    pub fn set_scale(&self, scale: Scale) {
        self.scale_bits
            .store(scale.bits() as u32, Ordering::Relaxed);
    }

    pub fn get_scale(&self) -> Scale {
        Scale::from_bits(self.scale_bits.load(Ordering::Relaxed) as u16)
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
        let mut out = Vec::with_capacity(self.target_pitch_contour.len());
        while let Some(p) = self.target_pitch_contour.pop() {
            out.push(p);
        }
        out
    }

    pub fn clear_target_pitch_contour(&self) {
        while self.target_pitch_contour.pop().is_some() {}
    }

    /// Enabled harmony voices: bit0 = 3rd, bit1 = 5th, bit2 = octave.
    pub fn set_harmony(&self, mask: u8) {
        self.harmony_mask.store(mask as u32, Ordering::Relaxed);
    }

    pub fn get_harmony(&self) -> u8 {
        self.harmony_mask.load(Ordering::Relaxed) as u8
    }

    /// Diatonic (in the selected key) vs absolute (fixed semitones).
    pub fn set_harmony_in_key(&self, in_key: bool) {
        self.harmony_in_key.store(in_key, Ordering::Relaxed);
    }

    /// Drain the produced-pitch log of `voice` (0 = main, 1..=3 harmonies;
    /// Hz per vocoder hop, 0 = silent) into `out`.
    pub fn drain_voice_pitch(&self, voice: usize, out: &mut Vec<f32>) {
        while let Some(f) = self.voice_pitch_log[voice].pop() {
            out.push(f);
        }
    }

    pub fn get_harmony_in_key(&self) -> bool {
        self.harmony_in_key.load(Ordering::Relaxed)
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
    /// Native default sampling rate (Hz), matching hardware.rs.
    const DEFAULT_SAMPLE_RATE: f32 = 44_100.0;
    /// Time constant of the correction-ratio smoothing filter. The per-hop
    /// one-pole coefficient is derived from this and the actual sample rate,
    /// so the smoothing speed is independent of rate and hop size. ~12.7ms
    /// reproduces the previous fixed alpha=0.6 at 44.1kHz / 512-sample hop.
    pub(crate) const SMOOTHING_TAU_SECONDS: f32 = 0.0127;

    pub fn new() -> Self {
        Self::with_scale(Scale::pentatonic(Note::C))
    }

    pub fn with_scale(scale: Scale) -> Self {
        Self::assemble(scale, Self::DEFAULT_SAMPLE_RATE)
    }

    /// Build a corrector whose pitch detector uses `sample_rate` (Hz). The
    /// web build must pass its real device rate here so detection is not
    /// skewed by the native-default rate.
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::with_scale_and_sample_rate(Scale::pentatonic(Note::C), sample_rate)
    }

    pub fn with_scale_and_sample_rate(scale: Scale, sample_rate: f32) -> Self {
        Self::assemble(scale, sample_rate)
    }

    fn assemble(scale: Scale, sample_rate: f32) -> Self {
        let yin = YinPitchDetector::with_sample_rate(sample_rate);
        // Per-hop one-pole coefficient from the time constant: the ratio_fn
        // runs once per hop (BUFFER_SIZE/4 samples), so alpha adapts to the
        // real hop period and keeps a constant smoothing time.
        let hop_period = HOP_SIZE as f32 / sample_rate;
        let smoothing_alpha = 1.0 - (-hop_period / Self::SMOOTHING_TAU_SECONDS).exp();

        // ~94 hops/sec (hop=BUFFER_SIZE/4 at 44.1-48kHz); this bounds the
        // target-pitch log to a few minutes of recording.
        const TARGET_CONTOUR_CAPACITY: usize = 32768;
        let controls = Arc::new(PitchCorrectorControls {
            shift_semitones: AtomicI32::new(Interval::UNISON.semitones()),
            scale_bits: AtomicU32::new(scale.bits() as u32),
            target_pitch_contour: ArrayQueue::new(TARGET_CONTOUR_CAPACITY),
            hop_detected_bits: AtomicU32::new(0),
            hop_target_semis: AtomicI32::new(NO_TARGET),
            harmony_mask: AtomicU32::new(0),
            voice_pitch_log: std::array::from_fn(|_| ArrayQueue::new(TARGET_CONTOUR_CAPACITY)),
            harmony_in_key: AtomicBool::new(true),
            contour: Mutex::new(Vec::new()),
            contour_hop: AtomicUsize::new(0),
        });

        let controls_clone = controls.clone();
        let detector = Mutex::new(yin);
        let snapper = Mutex::new(NoteSnapper::new());
        let smoothed_ratio = Mutex::new(1.0f32);
        let gap_hops = Mutex::new(0usize);
        let ratio_fn: RatioFn = Box::new(move |frame: &[f32]| {
            let shift_semitones = controls_clone.shift_semitones.load(Ordering::Relaxed);
            let shift_ratio = 2f32.powf(shift_semitones as f32 / 12.0);
            let detected = detector.lock().unwrap().detect(frame);

            // Check for active contour, otherwise snap to scale
            let target_pitch = {
                let contour = controls_clone.contour.lock().unwrap();
                if !contour.is_empty() {
                    let hop = controls_clone.contour_hop.fetch_add(1, Ordering::Relaxed);
                    contour[hop.min(contour.len() - 1)]
                } else {
                    let scale =
                        Scale::from_bits(controls_clone.scale_bits.load(Ordering::Relaxed) as u16);
                    detected.and_then(|freq| snapper.lock().unwrap().snap(freq, scale))
                }
            };

            // Lock-free, alloc-free log on the real-time audio thread. Drops
            // the entry if the bounded queue is full rather than allocating.
            let _ = controls_clone.target_pitch_contour.push(target_pitch);

            // Publish this hop's analysis for the harmony voices (parallel
            // pipelines fed in lockstep; they read these in their own
            // ratio_fns, which run after this one within the same sample).
            controls_clone.hop_detected_bits.store(
                detected.unwrap_or(0.0).to_bits(),
                Ordering::Relaxed,
            );
            controls_clone.hop_target_semis.store(
                target_pitch
                    .map(|p| p.semitones_from_c0() as i32 + shift_semitones)
                    .unwrap_or(NO_TARGET),
                Ordering::Relaxed,
            );

            let target_ratio = match (target_pitch, detected) {
                (Some(pitch), Some(freq)) => pitch.to_freq() / freq,
                _ => 1.0,
            };

            // Smooth the correction ratio to avoid abrupt changes. The
            // one-pole coefficient (smoothing_alpha) is derived from a fixed
            // time constant and the real sample rate/hop, so tracking speed
            // does not drift with the device rate.
            //
            // Only update when we have a valid detection; hold the previous
            // ratio during detection gaps (see f837ae5: decaying toward 1.0
            // every gap-hop caused a systematic downward bias).
            //
            // However, a *sustained* gap means the current note has ended,
            // so after GAP_RESET_HOPS of no detection we forget the held
            // ratio and return to neutral. This prevents the first hops of
            // an unrelated later note from being mis-corrected by a stale
            // ratio, without reintroducing the per-hop downward bias.
            const GAP_RESET_HOPS: usize = 43; // ~0.5s at 44.1kHz, hop=512
            let mut prev = smoothed_ratio.lock().unwrap();
            let mut gaps = gap_hops.lock().unwrap();
            if target_pitch.is_some() && detected.is_some() {
                *prev += smoothing_alpha * (target_ratio - *prev);
                *gaps = 0;
            } else {
                *gaps = gaps.saturating_add(1);
                if *gaps >= GAP_RESET_HOPS {
                    *prev = 1.0;
                }
            }
            let ratio = *prev * shift_ratio;
            // Log what this voice actually produces this hop (detected pitch
            // times the applied ratio; 0 when unvoiced).
            let _ = controls_clone.voice_pitch_log[0]
                .push(detected.map_or(0.0, |f| f * ratio));
            ratio
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

/// Harmony voice definitions: (label, scale degrees when in key, semitones
/// when absolute). An octave is +12 semitones in either mode.
const HARMONY_VOICES: [(usize, i32); 3] = [
    (2, 4),  // 3rd: 2 scale degrees / major third
    (4, 7),  // 5th: 4 scale degrees / perfect fifth
    (7, 12), // octave (degrees unused; +12 in both modes)
];
/// Mix level of each harmony voice relative to the main voice.
const HARMONY_GAIN: f32 = 0.5;

/// The main corrected voice plus up to three harmony voices, each its own
/// phase-vocoder pipeline running in lockstep on the same input. The main
/// corrector publishes its per-hop detection and target through the shared
/// controls; each harmony voice derives its own target from them — walking
/// the scale for diatonic harmonies or adding fixed semitones for absolute
/// ones — and the outputs are mixed with per-voice gains that fade in/out
/// (click-free) as voices are toggled or the signal goes unvoiced.
pub struct Harmonizer {
    main: PitchCorrector,
    voices: [PhaseVocoderPitchShifter<RatioFn>; 3],
    /// Per-voice smoothed mix gains (audio thread only; uncontended).
    gains: Mutex<[f32; 3]>,
    /// Per-sample one-pole coefficient for the gain fades (~5ms).
    gain_alpha: f32,
    controls: Arc<PitchCorrectorControls>,
}

impl Harmonizer {
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        let main = PitchCorrector::with_sample_rate(sample_rate);
        let controls = main.controls();

        let hop_period = HOP_SIZE as f32 / sample_rate;
        let smoothing_alpha =
            1.0 - (-hop_period / PitchCorrector::SMOOTHING_TAU_SECONDS).exp();

        let mut voice_idx = 0usize;
        let voices = HARMONY_VOICES.map(|(degrees, abs_semitones)| {
            voice_idx += 1;
            let log_idx = voice_idx; // 1..=3 ([0] is the main voice)
            let controls = controls.clone();
            let smoothed = Mutex::new(1.0f32);
            let gap_hops = Mutex::new(0usize);
            let ratio_fn: RatioFn = Box::new(move |_frame: &[f32]| {
                let detected = f32::from_bits(controls.hop_detected_bits.load(Ordering::Relaxed));
                let semis = controls.hop_target_semis.load(Ordering::Relaxed);
                let enabled = controls.harmony_mask.load(Ordering::Relaxed) & (1 << (log_idx - 1)) != 0;

                let target_freq = if detected > 0.0 && semis != NO_TARGET {
                    let base = Pitch::new(
                        Note::ALL[semis.rem_euclid(12) as usize],
                        semis.div_euclid(12) as i8,
                    );
                    let scale = controls.get_scale();
                    let in_key = controls.harmony_in_key.load(Ordering::Relaxed);
                    let harmony = if abs_semitones == 12 || !in_key || scale.is_empty() {
                        // Octave, absolute mode, or no key to walk in.
                        base.to_freq() * 2f32.powf(abs_semitones as f32 / 12.0)
                    } else {
                        scale.degree_above(base, degrees).to_freq()
                    };
                    Some(harmony)
                } else {
                    None
                };

                // Same smoothing + gap handling as the main voice.
                const GAP_RESET_HOPS: usize = 43;
                let mut prev = smoothed.lock().unwrap();
                let mut gaps = gap_hops.lock().unwrap();
                match target_freq {
                    Some(f) => {
                        *prev += smoothing_alpha * (f / detected - *prev);
                        *gaps = 0;
                    }
                    None => {
                        *gaps = gaps.saturating_add(1);
                        if *gaps >= GAP_RESET_HOPS {
                            *prev = 1.0;
                        }
                    }
                }
                // Log this voice's produced pitch for the plot: only while
                // enabled and voiced (it is faded out of the mix otherwise).
                let produced = if enabled && target_freq.is_some() {
                    detected * *prev
                } else {
                    0.0
                };
                let _ = controls.voice_pitch_log[log_idx].push(produced);
                *prev
            });
            PhaseVocoderPitchShifter::with_ratio_fn(ratio_fn)
        });

        Harmonizer {
            main,
            voices,
            gains: Mutex::new([0.0; 3]),
            gain_alpha: 1.0 - (-1.0 / (0.005 * sample_rate)).exp(),
            controls,
        }
    }

    pub fn controls(&self) -> Arc<PitchCorrectorControls> {
        self.controls.clone()
    }
}

impl StreamProcessor for Harmonizer {
    fn push_sample(&self, sample: f32) {
        // Main first: its hop publishes the analysis the voices read when
        // their (same-position) hop fires.
        self.main.push_sample(sample);
        for v in &self.voices {
            v.push_sample(sample);
        }
    }

    fn pop_sample(&self) -> Option<f32> {
        let main = self.main.pop_sample()?;
        let mask = self.controls.harmony_mask.load(Ordering::Relaxed);
        let voiced = self.controls.hop_target_semis.load(Ordering::Relaxed) != NO_TARGET;
        let mut gains = self.gains.lock().unwrap();
        let mut out = main;
        let mut norm = 1.0;
        for (i, v) in self.voices.iter().enumerate() {
            // Identical pipelines fed in lockstep emit in lockstep; treat a
            // (theoretical) miss as silence rather than stalling the mix.
            let h = v.pop_sample().unwrap_or(0.0);
            // Fade voices in only while enabled AND the signal is voiced —
            // harmonizing breath noise just doubles it.
            let target = if mask & (1 << i) != 0 && voiced { 1.0 } else { 0.0 };
            gains[i] += self.gain_alpha * (target - gains[i]);
            out += h * gains[i] * HARMONY_GAIN;
            norm += gains[i] * HARMONY_GAIN;
        }
        Some(out / norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::Pitch;
    use crate::signal_processing::BUFFER_SIZE;
    use std::f32::consts::TAU;

    const SAMPLE_RATE: usize = 44100;

    // --- Perf thresholds ---
    const PERF_CORRECTOR_TRANSPARENCY: f32 = 0.01; // max |similarity - 1.0|
    const PERF_SNAPPING_ACCURACY: f32 = 0.90; // min fraction on scale
    const PERF_TRACKING_PASS: f32 = 0.80; // min accuracy per rate
    const PERF_MIN_TRACKING_RATE: f32 = 3.0; // Hz (measured 4.0)
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
    fn shift_and_scale_roundtrip_through_atomics() {
        let corrector = PitchCorrector::new();
        let c = corrector.controls();
        for iv in [
            Interval::UNISON,
            Interval::PERFECT_FIFTH,
            Interval::PERFECT_FIFTH.negate(),
            Interval::OCTAVE,
            Interval::compound(SimpleInterval::MinorThird, -1),
        ] {
            c.set_shift(iv);
            assert_eq!(c.get_shift().semitones(), iv.semitones());
        }
        for s in [
            Scale::major(Note::C),
            Scale::minor_pentatonic(Note::A),
            Scale::chromatic(),
            Scale::empty(),
        ] {
            c.set_scale(s);
            assert_eq!(c.get_scale(), s);
        }
    }

    /// Spectrum check: with a harmony voice enabled, the output must contain
    /// energy at the harmony frequency — the diatonic third of A3 in C major
    /// is C4 (minor third), the absolute third is C#4 (major third). The two
    /// modes must land on their own note and not the other's.
    #[test]
    fn harmonizer_third_follows_the_mode() {
        use easyfft::dyn_size::realfft::DynRealFft;

        let bin_mag = |output: &[f32], freq: f32| -> f32 {
            let n = 16384;
            let tail = &output[output.len() - n..];
            let spectrum = tail.to_vec().real_fft();
            let bins = spectrum.get_frequency_bins();
            let bin = (freq * n as f32 / SAMPLE_RATE as f32).round() as usize;
            (bin - 1..=bin + 1).map(|b| bins[b].norm()).fold(0.0, f32::max)
        };

        let run = |in_key: bool| -> Vec<f32> {
            let h = Harmonizer::with_sample_rate(SAMPLE_RATE as f32);
            let c = h.controls();
            c.set_scale(Scale::major(Note::C));
            c.set_harmony(0b001); // 3rd only
            c.set_harmony_in_key(in_key);
            let freq = Pitch::new(Note::A, 3).to_freq(); // 220Hz, in C major
            let mut out = Vec::new();
            for i in 0..BUFFER_SIZE * 40 {
                h.push_sample((TAU * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.5);
                while let Some(s) = h.pop_sample() {
                    out.push(s);
                }
            }
            out
        };

        let c4 = Pitch::new(Note::C, 4).to_freq();
        let cs4 = Pitch::new(Note::CS, 4).to_freq();

        let diatonic = run(true);
        let d_c4 = bin_mag(&diatonic, c4);
        let d_cs4 = bin_mag(&diatonic, cs4);
        assert!(
            d_c4 > d_cs4 * 2.0,
            "in-key third of A3 should be C4, not C#4 (C4={d_c4:.1} C#4={d_cs4:.1})"
        );

        let absolute = run(false);
        let a_c4 = bin_mag(&absolute, c4);
        let a_cs4 = bin_mag(&absolute, cs4);
        assert!(
            a_cs4 > a_c4 * 2.0,
            "absolute third of A3 should be C#4, not C4 (C4={a_c4:.1} C#4={a_cs4:.1})"
        );

        // And the main voice is still there.
        let a3 = bin_mag(&diatonic, 220.0);
        assert!(a3 > d_c4 * 0.5, "main voice should remain dominant (A3={a3:.1})");
    }

    #[test]
    fn perf_pitch_corrector_no_alloc_after_warmup() {
        // The audio callback runs the corrector's ratio_fn every hop. It must
        // not allocate on the real-time thread, or long voiced passages cause
        // heap growth and glitches. A steady G3 keeps detection (and the
        // target-pitch contour path) active during the measured window.
        let corrector = PitchCorrector::new();
        let freq = 196.0; // G3

        let warmup = BUFFER_SIZE * 16;
        for i in 0..warmup {
            let s = (TAU * freq * i as f32 / SAMPLE_RATE as f32).sin();
            corrector.push_sample(s);
        }
        while corrector.pop_sample().is_some() {}

        assert_no_alloc::assert_no_alloc(|| {
            for i in 0..BUFFER_SIZE * 4 {
                let s = (TAU * freq * (warmup + i) as f32 / SAMPLE_RATE as f32).sin();
                corrector.push_sample(s);
            }
            while corrector.pop_sample().is_some() {}
        });
        eprintln!("[PERF] pitch_corrector_no_alloc: pass (threshold: zero allocations)");
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

        let num_samples = BUFFER_SIZE * 16;
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

        // Swing between G3 and A3 — adjacent pentatonic notes
        let g3 = Pitch::new(Note::G, 3).to_freq();
        let a3 = Pitch::new(Note::A, 3).to_freq();
        let center = (g3.ln() + a3.ln()) / 2.0;
        let swing = (a3.ln() - g3.ln()) / 2.0;

        // Vibrato rates 0.5Hz..4Hz in doublings. Rates >4Hz aren't needed for
        // the >=2Hz assertion (8Hz sat right on the pass threshold and was
        // flaky; 16Hz always failed), and each rate is a full extra signal to
        // process, so we stop at 4Hz.
        let rates: Vec<f32> = (0..4).map(|i| 0.5 * 2.0f32.powi(i)).collect();
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
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let pentatonic_c = Scale::pentatonic(Note::C);

        // Same vibrato as tracking test
        let g3 = Pitch::new(Note::G, 3).to_freq();
        let a3 = Pitch::new(Note::A, 3).to_freq();
        let center = (g3.ln() + a3.ln()) / 2.0;
        let swing = (a3.ln() - g3.ln()) / 2.0;

        // Noise levels as fraction of signal amplitude: 0.0..0.5. Levels above
        // ~0.5 fail detection entirely (YIN energy/CMND floor) and never affect
        // the pass check, so testing them was pure wasted work.
        let levels: Vec<f32> = (0..=5).map(|i| i as f32 * 0.1).collect();
        let samples_per_level = BUFFER_SIZE * 40;

        let mut best_noise = 0.0f32;
        // Fixed seed: keep the noise deterministic so the pass/fail threshold
        // is reproducible across runs and CI.
        let mut rng = StdRng::seed_from_u64(0x51173);

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
            } else if best_noise >= PERF_MIN_NOISE_TOLERANCE {
                // Tolerance already proven and accuracy is now falling with
                // more noise; higher levels won't pass, so stop early.
                break;
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
