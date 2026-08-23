use crate::complex_interpolation::ComplexInterpolate;
use crate::interpolation::Interpolate;
use crate::interpolation::InterpolationMethod;
use crossbeam_queue::ArrayQueue;
use easyfft::dyn_size::realfft::DynRealDft;
use easyfft::dyn_size::realfft::DynRealFft;
use easyfft::dyn_size::realfft::DynRealIfft;
use easyfft::num_complex::Complex;
use log::info;
use std::sync::Arc;
use std::sync::Mutex;

// Analysis window AND phase-vocoder frame size. This sets algorithmic
// latency (~one window: 2048/44.1kHz ≈ 46ms). It is deliberately large:
// YIN needs a full period in the window, so 2048 is what enables detection
// down to ~43Hz, and b84f0ba unified detection and synthesis onto this size
// so the input and target contours agree. Lowering it for latency would
// regress low-frequency detection; truly decoupling the two would require a
// separate, longer detection buffer feeding a shorter synthesis window.
pub const BUFFER_SIZE: usize = 2048;
/// Phase-vocoder hop: the DSP produces one analysis/synthesis frame (and
/// one per-voice pitch-log entry) every HOP_SIZE input samples.
pub const HOP_SIZE: usize = BUFFER_SIZE / 4;
pub const SPECTROGRAM_SIZE: usize = 8192;
// Native default sampling rate. The native path forces the device to this
// rate (see hardware.rs). YIN converts a detected period to Hz using a
// runtime rate (YinPitchDetector::with_sample_rate); this const is only the
// default for `new()` plus the spectrogram Nyquist and test signals. The web
// build runs at the browser AudioContext rate (often 48kHz) and threads that
// real rate through, so it must NOT rely on this const for detection.
const SAMPLE_RATE: usize = 44100;

// Peak-translation only tracks spectral peaks at or above this fraction of the
// frame's strongest peak. A tiny absolute floor let a moving tone's leakage /
// FM-sideband ripples register as peaks; each got translated with its own
// downshift-dependent offset and overwrote the main lobe's region, jittering
// and smearing a gliding pitch. Ignoring negligible peaks keeps the main lobe
// intact while leaving genuine harmonics (well above this floor) untouched.
const PEAK_REL_THRESHOLD: f32 = 1e-3;

/// A stateful stream transformer with single-owner semantics: the audio
/// thread owns the processor and drives it through `&mut self`. Sharing
/// (with the UI, or across callbacks) happens at the boundary via
/// lock-free queues and atomic cells — never inside a processor.
pub trait StreamProcessor {
    fn push_sample(&mut self, sample: f32);
    fn pop_sample(&mut self) -> Option<f32>;
}

pub trait BlockProcessor {
    fn process(&mut self, buffer: &mut [f32]);
}

pub trait FrequencyDomainBlockProcessor {
    fn process(&mut self, buffer: &mut DynRealDft<f32>);
}

pub struct NaivePitchShifter {
    scaling_ratio: f32,
}

pub struct HighPassFilter {
    frequency_response: DynRealDft<f32>,
}

pub struct LowPassFilter {
    frequency_response: DynRealDft<f32>,
}

pub struct FrequencyDomainPitchShifter {
    scaling_ratio: f32,
}

pub struct PhaseVocoderPitchShifter<F: FnMut(&[f32]) -> f32 + Send> {
    ratio_fn: F,
    hop_size: usize,
    input_buffer: ArrayQueue<f32>,
    output_buffer: ArrayQueue<f32>,
    state: PhaseVocoderState,
}

struct BinData {
    magnitudes: Vec<f32>,
    true_freq: Vec<f32>,
}

impl BinData {
    fn resize(&mut self, len: usize) {
        self.magnitudes.resize(len, 0.0);
        self.true_freq.resize(len, 0.0);
    }

    fn clear(&mut self) {
        self.magnitudes.iter_mut().for_each(|v| *v = 0.0);
        self.true_freq.iter_mut().for_each(|v| *v = 0.0);
    }
}

struct PhaseVocoderState {
    input_frame: Vec<f32>,
    input_pos: usize,
    prev_input_phase: Vec<f32>,
    prev_output_phase: Vec<f32>,
    output_accum: Vec<f32>,
    analysis: BinData,
    synthesis: BinData,
    synthesis_bins: Vec<Complex<f32>>,
    windowed: Vec<f32>,
    window: Vec<f32>,
    analysis_spectrum: Option<DynRealDft<f32>>,
    synthesis_spectrum: Option<DynRealDft<f32>>,
    ifft_output: Vec<f32>,
    // Pre-allocated phase-locking / peak-translation scratch
    new_output_phase: Vec<f32>,
    peaks: Vec<usize>,
}

pub struct DisplayProcessor<const I: usize = BUFFER_SIZE> {
    buffer: ArrayQueue<f32>,
    back_buffer: Box<[f32; I]>,
    front_buffer: Arc<Mutex<[f32; I]>>,
    write_index: usize,
}

pub struct OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    previous_clean_half_buffer: Box<[f32]>,
    previous_processed_half_buffer: Box<[f32]>,
    scratch: OlaBuffers,
    window: Box<[f32]>,
    block_processor: T,
}

struct OlaBuffers {
    first: Box<[f32; BUFFER_SIZE]>,
    second: Box<[f32; BUFFER_SIZE]>,
}

pub struct ComposedProcessor<F, S>
where
    F: StreamProcessor,
    S: StreamProcessor,
{
    first: F,
    second: S,
}

pub struct Segmenter<T>
where
    T: BlockProcessor,
{
    input_buffer: ArrayQueue<f32>,
    output_buffer: ArrayQueue<f32>,
    block_processor: T,
}

pub struct TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    frequency_domain_block_processor: T,
    spectrum: Option<DynRealDft<f32>>,
    ifft_buf: Vec<f32>,
}

impl<T> TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    pub fn new(frequency_domain_block_processor: T) -> Self {
        Self {
            frequency_domain_block_processor,
            spectrum: Some(DynRealDft::new(
                0.0,
                &vec![Complex::default(); BUFFER_SIZE / 2],
                BUFFER_SIZE,
            )),
            ifft_buf: vec![0.0; BUFFER_SIZE],
        }
    }
}

impl<T> BlockProcessor for TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    fn process(&mut self, buffer: &mut [f32]) {
        let spectrum = self.spectrum.as_mut().unwrap();
        buffer.real_fft_using(spectrum);
        self.frequency_domain_block_processor.process(spectrum);

        let ifft_buf = &mut self.ifft_buf;
        spectrum.real_ifft_using(ifft_buf);
        buffer.copy_from_slice(ifft_buf);
        for sample in buffer {
            *sample /= BUFFER_SIZE as f32;
        }
    }
}

impl<F, S> StreamProcessor for ComposedProcessor<F, S>
where
    F: StreamProcessor,
    S: StreamProcessor,
{
    fn push_sample(&mut self, sample: f32) {
        self.first.push_sample(sample);
        while let Some(sample) = self.first.pop_sample() {
            self.second.push_sample(sample);
        }
    }
    fn pop_sample(&mut self) -> Option<f32> {
        self.second.pop_sample()
    }
}

impl<F, S> ComposedProcessor<F, S>
where
    F: StreamProcessor,
    S: StreamProcessor,
{
    pub fn new(first: F, second: S) -> Self {
        info!("Creating new ComposedProcessor");
        Self { first, second }
    }
}

pub fn compose<F, S>(first: F, second: S) -> impl StreamProcessor
where
    F: StreamProcessor,
    S: StreamProcessor,
{
    ComposedProcessor::new(first, second)
}

impl<T> Segmenter<T>
where
    T: BlockProcessor,
{
    pub fn new(block_processor: T) -> Self {
        info!("Creating new Segmenter");
        Self {
            input_buffer: ArrayQueue::new(BUFFER_SIZE * 4),
            output_buffer: ArrayQueue::new(BUFFER_SIZE * 4),
            block_processor,
        }
    }
}

impl<T> StreamProcessor for Segmenter<T>
where
    T: BlockProcessor,
{
    fn pop_sample(&mut self) -> Option<f32> {
        self.output_buffer.pop()
    }

    fn push_sample(&mut self, sample: f32) {
        if self.input_buffer.push(sample).is_err() {
            log::warn!("Segmenter: input buffer overflow — dropping sample");
        }
        if self.input_buffer.len() > BUFFER_SIZE {
            let mut buffer = [0.0; BUFFER_SIZE];
            for sample in &mut buffer {
                *sample = self.input_buffer.pop().unwrap();
            }
            self.block_processor.process(&mut buffer);
            for sample in buffer {
                if self.output_buffer.push(sample).is_err() {
                    log::warn!("Segmenter: output buffer overflow — dropping sample");
                }
            }
        }
    }
}

impl<const I: usize> Default for DisplayProcessor<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const I: usize> DisplayProcessor<I> {
    pub fn new() -> Self {
        info!("Creating new DisplayProcessor of size {}", I);
        Self {
            buffer: ArrayQueue::new(I * 4),
            back_buffer: Box::new([0.0; I]),
            front_buffer: Arc::new(Mutex::new([0.0; I])),
            write_index: 0,
        }
    }

    pub fn clone_display_buffer(&self) -> Arc<Mutex<[f32; I]>> {
        self.front_buffer.clone()
    }
}

impl<const I: usize> StreamProcessor for DisplayProcessor<I> {
    fn push_sample(&mut self, sample: f32) {
        if self.buffer.push(sample).is_err() {
            log::warn!("DisplayProcessor: buffer overflow — dropping sample");
        }
    }

    fn pop_sample(&mut self) -> Option<f32> {
        let sample = self.buffer.pop()?;
        self.back_buffer[self.write_index] = sample;
        self.write_index += 1;
        if self.write_index >= I {
            // Back buffer full — swap to front (the UI's view; the one
            // genuine cross-thread boundary here).
            let mut front = self.front_buffer.lock().unwrap();
            front.copy_from_slice(self.back_buffer.as_ref());
            self.write_index = 0;
        }
        Some(sample)
    }
}

impl NaivePitchShifter {
    pub fn new(scaling_ratio: f32) -> Self {
        info!("Creating new NaivePitchShifter");
        Self { scaling_ratio }
    }
}

#[macro_export]
macro_rules! pipeline {
    ($first_processor:expr_2021$(,)?) => {
        $first_processor
    };

    ($first_processor:expr_2021, $($other_processors:expr_2021),+ $(,)?) => {
        $crate::signal_processing::compose(
            pipeline! {$first_processor},
            pipeline! { $($other_processors),+ },
        )
    };
}
pub use pipeline;

impl BlockProcessor for NaivePitchShifter {
    fn process(&mut self, buffer: &mut [f32]) {
        let mut output_buffer = [0.0; BUFFER_SIZE];
        for (index, sample) in output_buffer.iter_mut().enumerate() {
            *sample = (index as f32 * self.scaling_ratio) % (BUFFER_SIZE as f32 - 1.0);
        }
        buffer.interpolate_samples(&mut output_buffer, InterpolationMethod::Linear);
        buffer.copy_from_slice(&output_buffer);
    }
}

const fn frequency_to_bin(frequency: usize) -> usize {
    let highest_bin = BUFFER_SIZE / 2 + 1;
    let highest_frequency = SAMPLE_RATE / 2;
    highest_bin * frequency / highest_frequency
}

const fn get_cutoff_bin(frequency: usize) -> Option<usize> {
    let cutoff_bin = frequency_to_bin(frequency);

    if cutoff_bin > (BUFFER_SIZE / 2 + 1) {
        None
    } else {
        Some(cutoff_bin)
    }
}

impl HighPassFilter {
    pub fn new(cutoff_frequency: usize) -> Self {
        info!("Creating new HighPassFilter");
        let cutoff_bin = get_cutoff_bin(cutoff_frequency);
        let zeroth_bin = if cutoff_bin.is_some() { 0.0 } else { 1.0 };

        let mut frequency_bins = vec![Complex::default(); BUFFER_SIZE / 2];
        let frequency_bins = if let Some(cutoff_bin) = cutoff_bin {
            for bin in frequency_bins[cutoff_bin..].iter_mut() {
                *bin = Complex::new(1.0, 0.0);
            }
            frequency_bins.into_boxed_slice()
        } else {
            frequency_bins.into_boxed_slice()
        };

        let frequency_response = DynRealDft::new(zeroth_bin, &frequency_bins, BUFFER_SIZE);

        Self { frequency_response }
    }
}

impl FrequencyDomainBlockProcessor for HighPassFilter {
    fn process(&mut self, spectrum: &mut DynRealDft<f32>) {
        let resp_bins = self.frequency_response.get_frequency_bins();
        for (s, r) in spectrum.get_frequency_bins_mut().iter_mut().zip(resp_bins) {
            *s *= r;
        }
        *spectrum.get_offset_mut() *= self.frequency_response.get_offset();
    }
}

impl LowPassFilter {
    pub fn new(cutoff_frequency: usize) -> Self {
        info!("Creating new LowPassFilter");
        let cutoff_bin = get_cutoff_bin(cutoff_frequency);
        let zeroth_bin = 1.0;

        let mut frequency_bins = vec![Complex::default(); BUFFER_SIZE / 2];
        let frequency_bins = if let Some(cutoff_bin) = cutoff_bin {
            for bin in frequency_bins[..cutoff_bin].iter_mut() {
                *bin = Complex::new(1.0, 0.0);
            }
            frequency_bins.into_boxed_slice()
        } else {
            frequency_bins.into_boxed_slice()
        };

        let frequency_response = DynRealDft::new(zeroth_bin, &frequency_bins, BUFFER_SIZE);

        Self { frequency_response }
    }
}

impl FrequencyDomainBlockProcessor for LowPassFilter {
    fn process(&mut self, spectrum: &mut DynRealDft<f32>) {
        let resp_bins = self.frequency_response.get_frequency_bins();
        for (s, r) in spectrum.get_frequency_bins_mut().iter_mut().zip(resp_bins) {
            *s *= r;
        }
        *spectrum.get_offset_mut() *= self.frequency_response.get_offset();
    }
}

impl FrequencyDomainPitchShifter {
    pub fn new(scaling_ratio: f32) -> Self {
        info!("Creating new FrequencyDomainPitchShifter");
        Self { scaling_ratio }
    }
}

impl FrequencyDomainBlockProcessor for FrequencyDomainPitchShifter {
    fn process(&mut self, spectrum: &mut DynRealDft<f32>) {
        let interpolation_clone = spectrum.clone();

        spectrum
            .get_frequency_bins_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(index, bin)| {
                let index = index as f32 / self.scaling_ratio;
                *bin =
                    if index.ceil() >= (interpolation_clone.get_frequency_bins().len() - 1) as f32 {
                        Complex::default()
                    } else {
                        interpolation_clone
                            .get_frequency_bins()
                            .interpolate_sample(index)
                    }
            });
    }
}

impl PhaseVocoderPitchShifter<fn(&[f32]) -> f32> {
    pub fn new(scaling_ratio: f32) -> PhaseVocoderPitchShifter<impl FnMut(&[f32]) -> f32 + Send> {
        PhaseVocoderPitchShifter::with_ratio_fn(move |_: &[f32]| scaling_ratio)
    }
}

impl<F: FnMut(&[f32]) -> f32 + Send> PhaseVocoderPitchShifter<F> {
    pub fn with_ratio_fn(ratio_fn: F) -> Self {
        info!("Creating new PhaseVocoderPitchShifter with dynamic ratio");
        let hop_size = HOP_SIZE;
        let window: Vec<f32> = apodize::hanning_iter(BUFFER_SIZE)
            .map(|w| w as f32)
            .collect();
        Self {
            ratio_fn,
            hop_size,
            input_buffer: ArrayQueue::new(BUFFER_SIZE * 4),
            output_buffer: ArrayQueue::new(BUFFER_SIZE * 4),
            state: PhaseVocoderState {
                input_frame: vec![0.0; BUFFER_SIZE],
                input_pos: 0,
                prev_input_phase: vec![],
                prev_output_phase: vec![],
                output_accum: vec![0.0; BUFFER_SIZE],
                analysis: BinData {
                    magnitudes: vec![],
                    true_freq: vec![],
                },
                synthesis: BinData {
                    magnitudes: vec![],
                    true_freq: vec![],
                },
                synthesis_bins: vec![],
                windowed: vec![0.0; BUFFER_SIZE],
                window,
                analysis_spectrum: Some(DynRealDft::new(
                    0.0,
                    &vec![Complex::default(); BUFFER_SIZE / 2],
                    BUFFER_SIZE,
                )),
                synthesis_spectrum: Some(DynRealDft::new(
                    0.0,
                    &vec![Complex::default(); BUFFER_SIZE / 2],
                    BUFFER_SIZE,
                )),
                ifft_output: vec![0.0; BUFFER_SIZE],
                new_output_phase: vec![0.0; BUFFER_SIZE / 2],
                peaks: Vec::with_capacity(BUFFER_SIZE / 2),
            },
        }
    }

    fn process_frame(state: &mut PhaseVocoderState, scaling_ratio: f32, hop_size: usize) {
        let expected_phase_advance = |bin: usize| -> f32 {
            std::f32::consts::TAU * bin as f32 * hop_size as f32 / BUFFER_SIZE as f32
        };

        // Apply analysis window
        for (i, (s, w)) in state
            .input_frame
            .iter()
            .zip(state.window.iter())
            .enumerate()
        {
            state.windowed[i] = s * w;
        }

        // FFT
        state
            .windowed
            .real_fft_using(state.analysis_spectrum.as_mut().unwrap());
        let spectrum = state.analysis_spectrum.as_ref().unwrap();
        let bins = spectrum.get_frequency_bins();
        let num_bins = bins.len();

        // Resize scratch vectors on first call
        if state.prev_input_phase.len() != num_bins {
            state.prev_input_phase.resize(num_bins, 0.0);
            state.prev_output_phase.resize(num_bins, 0.0);
            state.new_output_phase.resize(num_bins, 0.0);
            state.analysis.resize(num_bins);
            state.synthesis.resize(num_bins);
            state
                .synthesis_bins
                .resize(BUFFER_SIZE / 2, Complex::default());
        }

        // Analysis: compute magnitude and time derivative (instantaneous freq)
        for (k, bin) in bins.iter().enumerate().take(num_bins) {
            state.analysis.magnitudes[k] = bin.norm();
            let phase = bin.arg();

            // Time derivative (backward difference)
            let mut phase_diff = phase - state.prev_input_phase[k];
            state.prev_input_phase[k] = phase;
            phase_diff -= expected_phase_advance(k + 1);
            phase_diff = phase_diff.rem_euclid(std::f32::consts::TAU);
            if phase_diff > std::f32::consts::PI {
                phase_diff -= std::f32::consts::TAU;
            }
            state.analysis.true_freq[k] = expected_phase_advance(k + 1) + phase_diff;
        }

        // Synthesis: shift bins
        state.synthesis.clear();
        for b in state.synthesis_bins.iter_mut() {
            *b = Complex::default();
        }

        // Pitch shift by peak translation (Laroche & Dolson, 1999). Resampling
        // the whole magnitude envelope (synthesis[k] = analysis[k/ratio])
        // stretches every spectral peak's main lobe by `scaling_ratio`, so a
        // shifted pure tone is no longer a single narrow lobe and radiates a
        // comb of hop-rate sidebands. Instead, translate each analysis peak's
        // region of influence *rigidly* to its shifted location, preserving the
        // lobe shape. Phase uses identity phase locking: the peak advances in
        // time at the shifted frequency and its region locks to it, keeping the
        // analysis inter-bin phase structure. `prev_input_phase` holds this
        // frame's analysis phase after the analysis loop above.

        // 1) Analysis-domain peaks: local maxima over ±2 bins.
        state.peaks.clear();
        let max_mag = state
            .analysis
            .magnitudes
            .iter()
            .take(num_bins)
            .cloned()
            .fold(0.0f32, f32::max);
        let peak_floor = max_mag * PEAK_REL_THRESHOLD;
        for a in 0..num_bins {
            let s = state.analysis.magnitudes[a];
            let is_peak = s > peak_floor
                && (a < 1 || s > state.analysis.magnitudes[a - 1])
                && (a < 2 || s > state.analysis.magnitudes[a - 2])
                && (a + 1 >= num_bins || s > state.analysis.magnitudes[a + 1])
                && (a + 2 >= num_bins || s > state.analysis.magnitudes[a + 2]);
            if is_peak {
                state.peaks.push(a);
            }
        }

        // Keep uncovered (silent) bins phase-continuous; covered bins are
        // overwritten below.
        state
            .new_output_phase
            .copy_from_slice(&state.prev_output_phase);

        // 2) Translate each peak's region of influence to its shifted location.
        let mut region_start = 0usize;
        for i in 0..state.peaks.len() {
            let a = state.peaks[i];
            let region_end = if i + 1 < state.peaks.len() {
                (a + state.peaks[i + 1]) / 2
            } else {
                num_bins - 1
            };

            // Shifted peak bin and its time-advanced synthesis phase. The phase
            // advances at the shifted instantaneous frequency (ratio * analysis
            // rate), so pitch is set by the phase even though the magnitude lobe
            // lands on the nearest integer bin.
            let s = ((a + 1) as f32 * scaling_ratio).round() as isize - 1;
            let peak_advance = state.analysis.true_freq[a] * scaling_ratio;
            let peak_phase = if s >= 0 && (s as usize) < num_bins {
                state.prev_output_phase[s as usize] + peak_advance
            } else {
                peak_advance
            };
            let peak_ana = state.prev_input_phase[a];

            // Rigidly copy the region, offset by (s - a); lock phases to peak.
            for b in region_start..=region_end {
                let t = s + (b as isize - a as isize);
                if t < 0 || t as usize >= num_bins {
                    continue;
                }
                let t = t as usize;
                state.synthesis.magnitudes[t] = state.analysis.magnitudes[b];
                state.new_output_phase[t] = peak_phase + (state.prev_input_phase[b] - peak_ana);
            }
            region_start = region_end + 1;
        }

        // Carry phase forward, then build the synthesis spectrum.
        state
            .prev_output_phase
            .copy_from_slice(&state.new_output_phase);
        for k in 0..num_bins {
            state.synthesis_bins[k] =
                Complex::from_polar(state.synthesis.magnitudes[k], state.new_output_phase[k]);
        }

        // IFFT
        {
            let synth = state.synthesis_spectrum.as_mut().unwrap();
            *synth.get_offset_mut() = *state.analysis_spectrum.as_ref().unwrap().get_offset();
            let n = synth.get_frequency_bins().len();
            synth
                .get_frequency_bins_mut()
                .copy_from_slice(&state.synthesis_bins[..n]);
        }
        state
            .synthesis_spectrum
            .as_ref()
            .unwrap()
            .real_ifft_using(&mut state.ifft_output);
        for s in state.ifft_output.iter_mut() {
            *s /= BUFFER_SIZE as f32;
        }

        // Apply synthesis window, normalized for 75% overlap (Hanning² sums to 1.5)
        for (s, w) in state.ifft_output.iter_mut().zip(state.window.iter()) {
            *s *= w / 1.5;
        }
    }
}

/// Spectral freeze: sustain the sound at one instant indefinitely.
///
/// Built from two overlapping analysis frames around a position in a
/// recording: per bin we keep the magnitude and the measured instantaneous
/// frequency (phase velocity), then synthesize hop after hop by advancing
/// each bin's phase at that velocity — the phase-vocoder synthesis step with
/// the analysis clock stopped. Pull samples with `next_sample()`; the OLA
/// windowing fades the first frame in from silence, so starting is
/// click-free by construction.
pub struct SpectralFreeze {
    magnitudes: Vec<f32>,
    /// Per-hop phase advance per bin (radians).
    phase_velocity: Vec<f32>,
    /// Current synthesis phase per bin.
    phases: Vec<f32>,
    window: Vec<f32>,
    spectrum: DynRealDft<f32>,
    ifft_output: Vec<f32>,
    /// Overlap-add accumulator; the front HOP_SIZE samples are ready.
    accum: Vec<f32>,
    /// Read cursor into the current hop's ready samples.
    cursor: usize,
}

impl SpectralFreeze {
    /// Analyze the `BUFFER_SIZE + HOP_SIZE` samples centered on `position`
    /// in `samples`. Returns `None` when there is not enough audio around
    /// the position.
    pub fn new(samples: &[f32], position: crate::units::SampleIdx) -> Option<Self> {
        let need = BUFFER_SIZE + HOP_SIZE;
        let start = position.0.checked_sub(need / 2)?;
        if start + need > samples.len() {
            return None;
        }
        let window: Vec<f32> = apodize::hanning_iter(BUFFER_SIZE)
            .map(|w| w as f32)
            .collect();

        let fft_at = |offset: usize| -> DynRealDft<f32> {
            let frame: Vec<f32> = samples[start + offset..start + offset + BUFFER_SIZE]
                .iter()
                .zip(&window)
                .map(|(s, w)| s * w)
                .collect();
            frame.real_fft()
        };
        let a = fft_at(0);
        let b = fft_at(HOP_SIZE);

        let bins_a = a.get_frequency_bins();
        let bins_b = b.get_frequency_bins();
        let num_bins = bins_a.len();
        let expected = |k: usize| {
            std::f32::consts::TAU * (k + 1) as f32 * HOP_SIZE as f32 / BUFFER_SIZE as f32
        };

        let mut magnitudes = vec![0.0; num_bins];
        let mut phase_velocity = vec![0.0; num_bins];
        let mut phases = vec![0.0; num_bins];
        for k in 0..num_bins {
            magnitudes[k] = bins_b[k].norm();
            phases[k] = bins_b[k].arg();
            // Instantaneous frequency: expected advance for this bin plus the
            // wrapped deviation actually measured between the two frames.
            let mut dev = bins_b[k].arg() - bins_a[k].arg() - expected(k);
            dev = dev.rem_euclid(std::f32::consts::TAU);
            if dev > std::f32::consts::PI {
                dev -= std::f32::consts::TAU;
            }
            phase_velocity[k] = expected(k) + dev;
        }

        let spectrum = a; // reuse as synthesis scratch (same size/offset kind)
        Some(SpectralFreeze {
            magnitudes,
            phase_velocity,
            phases,
            window,
            spectrum,
            ifft_output: vec![0.0; BUFFER_SIZE],
            accum: vec![0.0; BUFFER_SIZE],
            cursor: HOP_SIZE, // force a synthesis hop on the first pull
        })
    }

    /// Next sample of the sustained sound. Alloc-free.
    pub fn next_sample(&mut self) -> f32 {
        if self.cursor >= HOP_SIZE {
            self.synthesize_hop();
            self.cursor = 0;
        }
        let s = self.accum[self.cursor];
        self.cursor += 1;
        s
    }

    fn synthesize_hop(&mut self) {
        // Advance every bin's phase by one hop at its own velocity.
        for (p, v) in self.phases.iter_mut().zip(&self.phase_velocity) {
            *p = (*p + v).rem_euclid(std::f32::consts::TAU);
        }
        *self.spectrum.get_offset_mut() = 0.0;
        for (bin, (m, p)) in self
            .spectrum
            .get_frequency_bins_mut()
            .iter_mut()
            .zip(self.magnitudes.iter().zip(&self.phases))
        {
            *bin = Complex::from_polar(*m, *p);
        }
        self.spectrum.real_ifft_using(&mut self.ifft_output);

        // Shift the accumulator left by one hop and overlap-add the new
        // frame. The magnitudes already carry the analysis Hann; one more
        // synthesis Hann makes Hann² whose 75%-overlap sum is 1.5, exactly
        // like the vocoder (IFFT is unnormalized, hence / BUFFER_SIZE).
        self.accum.copy_within(HOP_SIZE.., 0);
        for s in &mut self.accum[BUFFER_SIZE - HOP_SIZE..] {
            *s = 0.0;
        }
        let norm = BUFFER_SIZE as f32 * 1.5;
        for (acc, (s, w)) in self
            .accum
            .iter_mut()
            .zip(self.ifft_output.iter().zip(&self.window))
        {
            *acc += s * w / norm;
        }
    }
}

impl<F: FnMut(&[f32]) -> f32 + Send> StreamProcessor for PhaseVocoderPitchShifter<F> {
    fn push_sample(&mut self, sample: f32) {
        if self.input_buffer.push(sample).is_err() {
            log::warn!("PhaseVocoder: input buffer overflow — dropping sample");
        }

        if self.input_buffer.len() >= self.hop_size {
            let state = &mut self.state;

            // Shift input frame left by hop_size
            state.input_frame.copy_within(self.hop_size.., 0);
            for i in (BUFFER_SIZE - self.hop_size)..BUFFER_SIZE {
                state.input_frame[i] = self.input_buffer.pop().unwrap();
            }

            state.input_pos += self.hop_size;
            if state.input_pos < BUFFER_SIZE {
                return;
            }
            state.input_pos = BUFFER_SIZE;

            let scaling_ratio = (self.ratio_fn)(&state.input_frame);
            Self::process_frame(state, scaling_ratio, self.hop_size);

            // Overlap-add into accumulator
            for i in 0..state.ifft_output.len() {
                state.output_accum[i] += state.ifft_output[i];
            }

            // Output hop_size samples
            for i in 0..self.hop_size {
                if self.output_buffer.push(state.output_accum[i]).is_err() {
                    log::warn!("PhaseVocoder: output buffer overflow — dropping sample");
                }
            }

            // Shift accumulator
            state.output_accum.copy_within(self.hop_size.., 0);
            for i in (BUFFER_SIZE - self.hop_size)..BUFFER_SIZE {
                state.output_accum[i] = 0.0;
            }
        }
    }

    fn pop_sample(&mut self) -> Option<f32> {
        self.output_buffer.pop()
    }
}

impl<T> BlockProcessor for OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    fn process(&mut self, buffer: &mut [f32]) {
        let scratch = &mut self.scratch;
        let previous_clean_half_buffer = &mut self.previous_clean_half_buffer;

        // Build first block: previous second half + current first half
        scratch.first[..BUFFER_SIZE / 2].copy_from_slice(previous_clean_half_buffer);
        scratch.first[BUFFER_SIZE / 2..].copy_from_slice(&buffer[..BUFFER_SIZE / 2]);

        // Build second block: current input buffer
        scratch.second.copy_from_slice(buffer);

        // Save second half of clean input for next call
        previous_clean_half_buffer.copy_from_slice(&buffer[BUFFER_SIZE / 2..]);

        // Process each block
        self.block_processor.process(&mut *scratch.first);
        self.block_processor.process(&mut *scratch.second);

        // Apply hanning window AFTER processing for smooth reconstruction
        for (sample, w) in scratch.first.iter_mut().zip(self.window.iter()) {
            *sample *= w;
        }
        for (sample, w) in scratch.second.iter_mut().zip(self.window.iter()) {
            *sample *= w;
        }

        // Overlap and add second half of first block and first half of second block
        for i in 0..BUFFER_SIZE / 2 {
            scratch.first[BUFFER_SIZE / 2 + i] += scratch.second[i];
        }

        // Overlap and add first half of first block with previous processed tail
        let previous_processed_half_buffer = &mut self.previous_processed_half_buffer;
        for i in 0..BUFFER_SIZE / 2 {
            scratch.first[i] += previous_processed_half_buffer[i];
        }

        // Save second half of second block for next call
        previous_processed_half_buffer.copy_from_slice(&scratch.second[BUFFER_SIZE / 2..]);

        buffer.copy_from_slice(&*scratch.first);
    }
}

impl<T> OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    #[allow(dead_code)]
    pub fn new(block_processor: T) -> Self {
        info!("Creating new OverlapAndAddProcessor");
        let window: Box<[f32]> = apodize::hanning_iter(BUFFER_SIZE)
            .map(|w| w as f32)
            .collect();
        Self {
            previous_clean_half_buffer: Box::new([0.0; BUFFER_SIZE / 2]),
            previous_processed_half_buffer: Box::new([0.0; BUFFER_SIZE / 2]),
            scratch: OlaBuffers {
                first: Box::new([0.0; BUFFER_SIZE]),
                second: Box::new([0.0; BUFFER_SIZE]),
            },
            window,
            block_processor,
        }
    }
}

const DEFAULT_YIN_THRESHOLD: f32 = 0.15;

pub struct YinPitchDetector {
    threshold: f32,
    cmnd: Vec<f32>,
    /// Sampling rate used to convert a detected period (in samples) to Hz.
    sample_rate: f32,
    /// Slowly-decaying estimate of recent peak frame energy, used by the
    /// adaptive gate so quiet-but-clear input is kept while inter-note
    /// silence is rejected relative to the current program level.
    peak_energy: f32,
    // --- Temporal state for voicing/pitch continuity ---
    /// Last detected frequency (0.0 if no recent detection).
    prev_frequency: f32,
    /// Number of consecutive voiced frames (saturates at 255).
    voiced_streak: u8,
    // --- FFT difference-function scratch (alloc-free after warmup) ---
    /// FFT length the scratch below is sized for (`2 * half_len`); 0 = unsized.
    fft_size: usize,
    /// First-half analysis window, zero-padded to `fft_size`.
    window_buf: Vec<f32>,
    /// Full `fft_size`-sample frame.
    signal_buf: Vec<f32>,
    /// Prefix sums of squared samples (`len = fft_size + 1`) for the energy
    /// terms of the difference function.
    psq: Vec<f32>,
    /// Real inverse-FFT output holding the cross-correlation (unnormalized).
    cc_buf: Vec<f32>,
    /// Spectrum of `window_buf`.
    window_spectrum: Option<DynRealDft<f32>>,
    /// Spectrum of `signal_buf`.
    signal_spectrum: Option<DynRealDft<f32>>,
    /// `conj(window_spectrum) * signal_spectrum`, inverse-transformed to `cc_buf`.
    corr_spectrum: Option<DynRealDft<f32>>,
}

impl Default for YinPitchDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl YinPitchDetector {
    /// Create a detector assuming the native `SAMPLE_RATE` (44.1kHz).
    pub fn new() -> Self {
        Self::with_sample_rate(SAMPLE_RATE as f32)
    }

    /// Create a detector for a specific sampling rate. The web build runs
    /// at the browser AudioContext rate (often 48kHz), so it must pass the
    /// real device rate here or detected pitch is skewed.
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self {
            threshold: DEFAULT_YIN_THRESHOLD,
            cmnd: vec![0.0; BUFFER_SIZE / 2],
            sample_rate,
            peak_energy: 0.0,
            prev_frequency: 0.0,
            voiced_streak: 0,
            fft_size: 0,
            window_buf: Vec::new(),
            signal_buf: Vec::new(),
            psq: Vec::new(),
            cc_buf: Vec::new(),
            window_spectrum: None,
            signal_spectrum: None,
            corr_spectrum: None,
        }
    }

    pub fn detect(&mut self, buffer: &[f32]) -> Option<f32> {
        let half_len = buffer.len() / 2;
        if half_len < 2 {
            return None;
        }

        // Adaptive energy gate. A fixed absolute threshold is tied to input
        // gain: it rejects quiet singing outright and cliffs hard in noise.
        // Instead, track a slowly-decaying peak energy and reject a frame
        // only when it is either true silence (denormal/DC floor) or far
        // below the recent program level (inter-note silence).
        let energy: f32 = buffer.iter().map(|s| s * s).sum::<f32>() / buffer.len() as f32;

        // Absolute floor for genuine silence, independent of program level.
        const SILENCE_FLOOR: f32 = 1e-6;
        // Relative floor: ~30 dB (energy 1e-3) below the recent peak.
        const RELATIVE_FLOOR: f32 = 1e-3;
        // Decay per call so the peak reflects the last ~second of audio.
        const PEAK_DECAY: f32 = 0.995;

        self.peak_energy = (self.peak_energy * PEAK_DECAY).max(energy);
        if energy < SILENCE_FLOOR || energy < self.peak_energy * RELATIVE_FLOOR {
            self.voiced_streak = 0;
            return None;
        }

        self.cmnd.resize(half_len, 0.0);
        self.cumulative_mean_normalized_difference(buffer, half_len);

        // Try standard threshold first.
        let mut tau = self.absolute_threshold();

        // Voicing continuity: if no dip below the standard threshold but
        // we've been voiced for several frames, try a relaxed threshold.
        // This bridges brief gaps where CMND doesn't quite dip below 0.15
        // but the signal is clearly still voiced.
        if tau.is_none() && self.voiced_streak >= 2 {
            tau = self.absolute_threshold_relaxed();
        }

        let tau = match tau {
            Some(t) => t,
            None => {
                self.voiced_streak = 0;
                return None;
            }
        };

        let refined_tau = parabolic_interpolation(&self.cmnd, tau);
        let frequency = self.sample_rate / refined_tau;

        if !(50.0..=4000.0).contains(&frequency) {
            self.voiced_streak = 0;
            return None;
        }

        // Update state.
        self.prev_frequency = frequency;
        self.voiced_streak = self.voiced_streak.saturating_add(1);

        Some(frequency)
    }

    /// Ensure the FFT scratch is sized for an `n`-point transform. Reallocates
    /// only when `n` changes, so a steady frame size is alloc-free after warmup.
    fn ensure_fft_scratch(&mut self, n: usize) {
        if self.fft_size == n {
            return;
        }
        self.window_buf = vec![0.0; n];
        self.signal_buf = vec![0.0; n];
        self.psq = vec![0.0; n + 1];
        self.cc_buf = vec![0.0; n];
        self.window_spectrum = Some(DynRealDft::new(0.0, &vec![Complex::default(); n / 2], n));
        self.signal_spectrum = Some(DynRealDft::new(0.0, &vec![Complex::default(); n / 2], n));
        self.corr_spectrum = Some(DynRealDft::new(0.0, &vec![Complex::default(); n / 2], n));
        self.fft_size = n;
    }

    /// Cumulative mean normalized difference function (YIN eq. 6-8).
    ///
    /// The difference function `d(τ) = Σ_{i<W} (x[i] - x[i+τ])²` (W = `half_len`)
    /// expands to `A + B(τ) - 2·C(τ)` where `A = Σ_{i<W} x[i]²` is constant,
    /// `B(τ) = Σ_{i<W} x[i+τ]²` is a sliding window energy (prefix sums), and
    /// `C(τ) = Σ_{i<W} x[i]·x[i+τ]` is a cross-correlation. Computing `C` via
    /// FFT (Wiener–Khinchin) turns the naive O(n²) double loop into O(n log n)
    /// while producing the same difference function.
    fn cumulative_mean_normalized_difference(&mut self, buffer: &[f32], half_len: usize) {
        let n = 2 * half_len;
        self.ensure_fft_scratch(n);

        // a = first-half window zero-padded to n; b = the full n-sample frame.
        // Zero-padding a to length n means the circular correlation at lag τ<W
        // never wraps (max index (W-1)+(W-1) = 2W-2 < n), so it equals the
        // linear cross-correlation C(τ) our direct loop computes.
        self.window_buf[..half_len].copy_from_slice(&buffer[..half_len]);
        self.window_buf[half_len..].fill(0.0);
        self.signal_buf.copy_from_slice(&buffer[..n]);

        // Prefix sums of squares for A and the sliding energy B(τ).
        self.psq[0] = 0.0;
        for i in 0..n {
            self.psq[i + 1] = self.psq[i] + self.signal_buf[i] * self.signal_buf[i];
        }

        // C(τ) = real_ifft(conj(FFT(a)) · FFT(b)) / n. easyfft's inverse is
        // unnormalized (like the phase vocoder, which divides by BUFFER_SIZE),
        // hence the 1/n below.
        self.window_buf
            .real_fft_using(self.window_spectrum.as_mut().unwrap());
        self.signal_buf
            .real_fft_using(self.signal_spectrum.as_mut().unwrap());
        {
            let a_spec = self.window_spectrum.as_ref().unwrap();
            let b_spec = self.signal_spectrum.as_ref().unwrap();
            let dc = a_spec.get_offset() * b_spec.get_offset();
            let corr = self.corr_spectrum.as_mut().unwrap();
            *corr.get_offset_mut() = dc;
            for (p, (av, bv)) in corr.get_frequency_bins_mut().iter_mut().zip(
                a_spec
                    .get_frequency_bins()
                    .iter()
                    .zip(b_spec.get_frequency_bins()),
            ) {
                *p = av.conj() * bv;
            }
        }
        self.corr_spectrum
            .as_mut()
            .unwrap()
            .real_ifft_using(&mut self.cc_buf);

        // d(τ) = A + B(τ) - 2·C(τ), then the cumulative mean normalization.
        self.cmnd[0] = 1.0;
        let a_energy = self.psq[half_len];
        let inv_n = 1.0 / n as f32;
        let mut running_sum = 0.0f32;
        for tau in 1..half_len {
            let b_energy = self.psq[tau + half_len] - self.psq[tau];
            let c = self.cc_buf[tau] * inv_n;
            // True d is ≥ 0; clamp tiny negatives from FFT round-off so the
            // running sum and CMND stay well-defined at periodic dips.
            let diff = (a_energy + b_energy - 2.0 * c).max(0.0);
            running_sum += diff;
            self.cmnd[tau] = if running_sum > 0.0 {
                diff * tau as f32 / running_sum
            } else {
                1.0
            };
        }
    }

    fn absolute_threshold(&self) -> Option<usize> {
        // Restrict tau search to a valid frequency range.
        // min_tau = sr/fmax (highest freq), max_tau = sr/fmin (lowest freq).
        // These match the post-detection range check in detect().
        let fmin = 50.0_f32;
        let fmax = 4000.0_f32;
        let min_tau = (self.sample_rate / fmax).floor() as usize;
        let max_tau = ((self.sample_rate / fmin).ceil() as usize).min(self.cmnd.len() - 1);

        // Standard YIN: find first tau below threshold, walk to local min.
        // By starting at min_tau (not tau=2), we avoid spurious dips at
        // very high frequencies that don't correspond to real fundamentals.
        for tau in min_tau.max(2)..max_tau {
            if self.cmnd[tau] < self.threshold {
                let mut best = tau;
                while best + 1 < max_tau && self.cmnd[best + 1] < self.cmnd[best] {
                    best += 1;
                }
                return Some(best);
            }
        }
        None
    }

    /// Relaxed threshold search for voicing continuity.
    /// Uses a higher threshold (more permissive) to bridge brief CMND
    /// dips that don't quite reach the standard threshold. Only called
    /// when we have recent voicing history (voiced_streak >= 3).
    fn absolute_threshold_relaxed(&self) -> Option<usize> {
        const RELAXED_THRESHOLD: f32 = 0.35;
        let fmin = 50.0_f32;
        let fmax = 4000.0_f32;
        let min_tau = (self.sample_rate / fmax).floor() as usize;
        let max_tau = ((self.sample_rate / fmin).ceil() as usize).min(self.cmnd.len() - 1);

        // If we have pitch history, narrow the search to ±1 octave of
        // the previous frequency to avoid jumping to a wrong candidate.
        let (search_min, search_max) = if self.prev_frequency > 0.0 {
            let prev_tau = self.sample_rate / self.prev_frequency;
            // ±1 octave = tau * 0.5 to tau * 2.0
            let lo = (prev_tau * 0.5) as usize;
            let hi = (prev_tau * 2.0) as usize;
            (lo.max(min_tau), hi.min(max_tau))
        } else {
            (min_tau, max_tau)
        };

        for tau in search_min.max(2)..search_max {
            if self.cmnd[tau] < RELAXED_THRESHOLD {
                let mut best = tau;
                while best + 1 < search_max && self.cmnd[best + 1] < self.cmnd[best] {
                    best += 1;
                }
                return Some(best);
            }
        }
        None
    }
}

fn parabolic_interpolation(cmnd: &[f32], tau: usize) -> f32 {
    if tau < 1 || tau >= cmnd.len() - 1 {
        return tau as f32;
    }
    let alpha = cmnd[tau - 1];
    let beta = cmnd[tau];
    let gamma = cmnd[tau + 1];
    // Guard the degenerate (flat/collinear) case: a zero denominator would
    // produce NaN/inf and silently discard an otherwise valid detection.
    let denom = alpha - 2.0 * beta + gamma;
    if denom.abs() < f32::EPSILON {
        return tau as f32;
    }
    let peak = 0.5 * (alpha - gamma) / denom;
    tau as f32 + peak
}
#[cfg(test)]
mod tests {

    #[test]
    fn spectral_freeze_sustains_the_frame() {
        let sr = 44_100.0;
        let freq = 220.0;
        // A tone with a couple of harmonics, like a voice.
        let samples: Vec<f32> = (0..sr as usize)
            .map(|i| {
                let t = i as f32 / sr;
                (std::f32::consts::TAU * freq * t).sin() * 0.4
                    + (std::f32::consts::TAU * 2.0 * freq * t).sin() * 0.15
            })
            .collect();

        let mut freeze =
            SpectralFreeze::new(&samples, crate::units::SampleIdx(samples.len() / 2)).unwrap();
        // Skip the OLA fade-in, then take two seconds of sustain.
        for _ in 0..BUFFER_SIZE {
            freeze.next_sample();
        }
        let mut out = vec![0.0f32; 2 * sr as usize];
        assert_no_alloc::assert_no_alloc(|| {
            for s in out.iter_mut() {
                *s = freeze.next_sample();
            }
        });

        // Pitch is preserved.
        let mut yin = YinPitchDetector::new();
        let detected = yin.detect(&out[..BUFFER_SIZE]).expect("freeze is voiced");
        assert!(
            (detected - freq).abs() < 3.0,
            "freeze should sustain {freq}Hz, detected {detected}Hz"
        );

        // Amplitude is stable: RMS of the first and last half agree, and is
        // in the ballpark of the source (windowing/OLA scaling correct).
        let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
        let (a, b) = out.split_at(out.len() / 2);
        let (ra, rb) = (rms(a), rms(b));
        let source_rms = rms(&samples);
        assert!(
            (ra / rb - 1.0).abs() < 0.05,
            "sustain should not decay or grow (first {ra:.4} vs last {rb:.4})"
        );
        assert!(
            ra > source_rms * 0.5 && ra < source_rms * 2.0,
            "freeze level should be near the source ({ra:.4} vs {source_rms:.4})"
        );

        // Not enough context near the edges: refuse, don't panic.
        assert!(SpectralFreeze::new(&samples, crate::units::SampleIdx(10)).is_none());
        assert!(
            SpectralFreeze::new(&samples, crate::units::SampleIdx(samples.len() - 10)).is_none()
        );
    }
    use super::*;

    const TEST_SAMPLE_SIZE: usize = BUFFER_SIZE * 10;

    // --- Perf thresholds ---
    const PERF_VOCODER_TRANSPARENCY: f32 = 0.01; // max |similarity - 1.0|
    const PERF_SPECTRAL_PURITY: f32 = 0.99; // min energy concentration
    const PERF_TRANSITION_WORST: f32 = 0.98; // min purity at transition
    const PERF_TRANSITION_AVG: f32 = 0.99; // min avg purity at transition
    const PERF_YIN_MEAN_CENTS: f32 = 0.5; // max mean pitch error (measured 0.16)
    const PERF_YIN_WORST_CENTS: f32 = 1.5; // max worst pitch error (measured 0.99)
    const PERF_FINE_SHIFT_CENTS: f32 = 1.0; // max realized error for sub-semitone shifts (measured 0.67)
    const PERF_VOICE_RPA: f32 = 0.95; // min raw pitch accuracy on synthetic voice (measured 100%)
    const PERF_VOICE_FPE_CENTS: f32 = 6.0; // max fine pitch error on synthetic voice (measured ~4-5)
    const TEST_EQUALITY_EPISLON: f32 = 0.002;

    struct PassthroughBlockProcessor;

    impl BlockProcessor for PassthroughBlockProcessor {
        fn process(&mut self, _buffer: &mut [f32]) {
            // Do nothing to buffer
        }
    }

    struct AmplitudeHalvingBlockProcessor;

    impl BlockProcessor for AmplitudeHalvingBlockProcessor {
        fn process(&mut self, buffer: &mut [f32]) {
            for sample in buffer.iter_mut() {
                *sample /= 2.0;
            }
        }
    }

    #[test]
    fn overlap_and_add_processor_is_transparent() {
        let mut passthrough_stream_processor =
            Segmenter::new(OverlapAndAddProcessor::new(PassthroughBlockProcessor));
        let queue = ArrayQueue::new(BUFFER_SIZE * 4);
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>();
            passthrough_stream_processor.push_sample(x);
            let _ = queue.push(x);
        }

        // Get rid of transients
        for _ in 0..BUFFER_SIZE {
            let _ = passthrough_stream_processor.pop_sample().unwrap();
            let _ = queue.pop().unwrap();
        }

        // Remove delay from OverlapAndAddProcessor
        for _ in 0..BUFFER_SIZE / 2 {
            let _ = passthrough_stream_processor.pop_sample().unwrap();
        }

        while let (Some(stream_processor_value), Some(queue_value)) =
            (passthrough_stream_processor.pop_sample(), queue.pop())
        {
            approx::assert_abs_diff_eq!(
                stream_processor_value,
                queue_value,
                epsilon = TEST_EQUALITY_EPISLON
            );
        }
    }

    #[test]
    fn overlap_and_add_processor_and_amplitude_halver_works_as_expected() {
        let mut passthrough_stream_processor =
            Segmenter::new(OverlapAndAddProcessor::new(AmplitudeHalvingBlockProcessor));
        let queue = ArrayQueue::new(BUFFER_SIZE * 4);
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>();
            passthrough_stream_processor.push_sample(x);
            let _ = queue.push(x);
        }

        // Get rid of transients
        for _ in 0..BUFFER_SIZE {
            let _ = passthrough_stream_processor.pop_sample().unwrap();
            let _ = queue.pop().unwrap();
        }

        // Remove delay from OverlapAndAddProcessor
        for _ in 0..BUFFER_SIZE / 2 {
            let _ = passthrough_stream_processor.pop_sample().unwrap();
        }

        while let (Some(stream_processor_value), Some(queue_value)) =
            (passthrough_stream_processor.pop_sample(), queue.pop())
        {
            approx::assert_abs_diff_eq!(
                stream_processor_value,
                queue_value / 2.0,
                epsilon = TEST_EQUALITY_EPISLON
            );
        }
    }

    #[test]
    fn apodize_hanning_window_sums_to_one() {
        let mut window_1: Vec<_> = apodize::hanning_iter(BUFFER_SIZE).collect();
        let window_2: Vec<_> = apodize::hanning_iter(BUFFER_SIZE).collect();

        for (w1, w2) in window_1[..BUFFER_SIZE / 2]
            .iter_mut()
            .zip(window_2[BUFFER_SIZE / 2..].iter())
        {
            *w1 += w2;
        }
        for sample in window_1[..BUFFER_SIZE / 2].iter() {
            approx::assert_abs_diff_eq!(*sample, 1.0, epsilon = TEST_EQUALITY_EPISLON as f64);
        }
    }

    #[test]
    fn segmenter_is_transparent() {
        let mut passthrough_stream_processor = Segmenter::new(PassthroughBlockProcessor);
        let queue = ArrayQueue::new(BUFFER_SIZE * 4);
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>();
            passthrough_stream_processor.push_sample(x);
            let _ = queue.push(x);
        }

        while let Some(stream_sample) = passthrough_stream_processor.pop_sample() {
            assert_eq!(stream_sample, queue.pop().unwrap());
        }
    }

    #[test]
    fn low_pass_filter_no_discontinuities_with_ola() {
        let freq = 100.0;
        let mut processor = Segmenter::new(OverlapAndAddProcessor::new(
            TimeToFrequencyDomainBlockProcessorConverter::new(LowPassFilter::new(440)),
        ));

        let num_samples = BUFFER_SIZE * 10;
        for i in 0..num_samples {
            let sample = (std::f32::consts::TAU * freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
        }

        for _ in 0..BUFFER_SIZE * 2 {
            let _ = processor.pop_sample();
        }

        let mut output = Vec::new();
        while let Some(s) = processor.pop_sample() {
            output.push(s);
        }

        let max_expected_delta = 0.05;
        let mut max_delta: f32 = 0.0;
        for window in output.windows(2) {
            let delta = (window[1] - window[0]).abs();
            max_delta = max_delta.max(delta);
        }

        // With OLA, block boundaries are smooth
        assert!(
            max_delta < max_expected_delta,
            "Discontinuity detected: max delta {max_delta} exceeds {max_expected_delta}"
        );
    }

    #[test]
    fn low_pass_filter_preserves_dc_component() {
        let cutoff = 440;
        let filter = LowPassFilter::new(cutoff);

        // Feed a signal with DC offset through the filter
        let mut buffer: [f32; BUFFER_SIZE] = [0.5; BUFFER_SIZE];
        let mut converter = TimeToFrequencyDomainBlockProcessorConverter::new(filter);
        converter.process(&mut buffer);

        let mean: f32 = buffer.iter().sum::<f32>() / buffer.len() as f32;

        // DC should be preserved by a low-pass filter
        assert!(
            (mean - 0.5).abs() < 0.01,
            "DC should be preserved, but mean was {mean}"
        );
    }

    #[test]
    fn frequency_domain_pitch_shifter_no_distortion() {
        let input_freq = 440.0;
        let expected_freq = input_freq * 0.5;
        let mut processor = Segmenter::new(OverlapAndAddProcessor::new(
            TimeToFrequencyDomainBlockProcessorConverter::new(FrequencyDomainPitchShifter::new(
                0.5,
            )),
        ));

        let num_samples = BUFFER_SIZE * 10;
        for i in 0..num_samples {
            let sample = (std::f32::consts::TAU * input_freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
        }

        // Skip transients
        for _ in 0..BUFFER_SIZE * 3 {
            let _ = processor.pop_sample();
        }

        // Collect output and check for discontinuities
        let mut output = Vec::new();
        while let Some(s) = processor.pop_sample() {
            output.push(s);
        }

        // Max delta for a sine at expected_freq (220Hz) at 44100Hz
        // is sin(TAU * 220 / 44100) ≈ 0.031
        let max_expected_delta = 0.1;
        let mut max_delta: f32 = 0.0;
        for window in output.windows(2) {
            let delta = (window[1] - window[0]).abs();
            max_delta = max_delta.max(delta);
        }

        assert!(
            max_delta < max_expected_delta,
            "Discontinuity detected: max delta {max_delta} exceeds {max_expected_delta}"
        );

        // Verify output has energy at expected frequency
        let mut block = [0.0f32; BUFFER_SIZE];
        block.copy_from_slice(&output[..BUFFER_SIZE]);
        let spectrum = block.real_fft();
        let bins = spectrum.get_frequency_bins();

        let expected_bin = frequency_to_bin(expected_freq as usize);
        let peak_bin = bins
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .unwrap()
            .0;

        assert!(
            (peak_bin as i32 - expected_bin as i32).unsigned_abs() <= 2,
            "Expected peak near bin {expected_bin} ({}Hz), got bin {peak_bin}",
            expected_freq
        );
    }

    #[test]
    fn frequency_domain_pitch_shifter_up_no_distortion() {
        let input_freq = 220.0;
        let scaling_ratio = 2.0;
        let expected_freq = input_freq * scaling_ratio;
        let mut processor = Segmenter::new(OverlapAndAddProcessor::new(
            TimeToFrequencyDomainBlockProcessorConverter::new(FrequencyDomainPitchShifter::new(
                scaling_ratio,
            )),
        ));

        let num_samples = BUFFER_SIZE * 10;
        for i in 0..num_samples {
            let sample = (std::f32::consts::TAU * input_freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
        }

        for _ in 0..BUFFER_SIZE * 3 {
            let _ = processor.pop_sample();
        }

        let mut output = Vec::new();
        while let Some(s) = processor.pop_sample() {
            output.push(s);
        }

        let max_expected_delta = 0.15;
        let mut max_delta: f32 = 0.0;
        for window in output.windows(2) {
            let delta = (window[1] - window[0]).abs();
            max_delta = max_delta.max(delta);
        }

        assert!(
            max_delta < max_expected_delta,
            "Discontinuity detected: max delta {max_delta} exceeds {max_expected_delta}"
        );

        let mut block = [0.0f32; BUFFER_SIZE];
        block.copy_from_slice(&output[..BUFFER_SIZE]);
        let spectrum = block.real_fft();
        let bins = spectrum.get_frequency_bins();

        let expected_bin = frequency_to_bin(expected_freq as usize);
        let peak_bin = bins
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .unwrap()
            .0;

        assert!(
            (peak_bin as i32 - expected_bin as i32).unsigned_abs() <= 2,
            "Expected peak near bin {expected_bin} ({}Hz), got bin {peak_bin}",
            expected_freq
        );
    }

    #[test]
    fn perf_phase_vocoder_unity_ratio_is_transparent() {
        let input_freq = 440.0;
        let mut processor = PhaseVocoderPitchShifter::new(1.0);

        let num_samples = BUFFER_SIZE * 40;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (std::f32::consts::TAU * input_freq * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        let mut output = Vec::new();
        for &s in &input {
            processor.push_sample(s);
            while let Some(o) = processor.pop_sample() {
                output.push(o);
            }
        }

        // Skip transients — use last portion of output
        let compare_len = BUFFER_SIZE * 5;
        assert!(
            output.len() > compare_len * 2,
            "Not enough output: {}",
            output.len()
        );
        let output_slice = &output[output.len() - compare_len..];
        // Align input: output is delayed by ~BUFFER_SIZE samples
        let delay = input.len() - output.len();
        let input_start = input.len() - compare_len - delay;
        let input_slice = &input[input_start..input_start + compare_len];

        // Cross-correlation at zero lag should be close to autocorrelation
        let cross: f32 = input_slice
            .iter()
            .zip(output_slice.iter())
            .map(|(a, b)| a * b)
            .sum();
        let auto: f32 = input_slice.iter().map(|a| a * a).sum();

        let similarity = cross / auto;
        eprintln!(
            "[PERF] phase_vocoder_unity_transparency: similarity={similarity:.4} (threshold: >{:.2})",
            1.0 - PERF_VOCODER_TRANSPARENCY
        );
        assert!(
            (similarity - 1.0).abs() < PERF_VOCODER_TRANSPARENCY,
            "Phase vocoder at ratio 1.0 should be transparent, but similarity was {similarity:.3}"
        );
    }

    #[test]
    fn phase_vocoder_pitch_shifter_produces_output() {
        let input_freq = 440.0;
        let mut processor = PhaseVocoderPitchShifter::new(0.5);

        let num_samples = BUFFER_SIZE * 10;
        for i in 0..num_samples {
            let sample = (std::f32::consts::TAU * input_freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
        }

        let mut output = Vec::new();
        while let Some(s) = processor.pop_sample() {
            output.push(s);
        }

        assert!(
            output.len() > BUFFER_SIZE,
            "Expected output samples, got {}",
            output.len()
        );

        // Check output isn't silence
        let max_amp: f32 = output.iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(
            max_amp > 0.01,
            "Output appears silent, max amplitude: {max_amp}"
        );
    }

    fn generate_sine(freq: f32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    #[test]
    fn yin_fft_difference_matches_direct_loop() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        // A voiced-like signal: a couple of harmonics plus noise, so the
        // difference function has real structure (dips) to compare.
        let mut rng = StdRng::seed_from_u64(0x000F_179E);
        let half_len = BUFFER_SIZE / 2;
        let n = 2 * half_len;
        let tau_hz = std::f32::consts::TAU;
        let buffer: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                0.5 * (tau_hz * 220.0 * t).sin()
                    + 0.25 * (tau_hz * 440.0 * t).sin()
                    + 0.05 * (rng.random::<f32>() * 2.0 - 1.0)
            })
            .collect();

        // Reference: the naive O(n^2) difference + CMND this replaced.
        let mut direct = vec![0.0f32; half_len];
        direct[0] = 1.0;
        let mut running = 0.0f32;
        for tau in 1..half_len {
            let mut diff = 0.0f32;
            for i in 0..half_len {
                let delta = buffer[i] - buffer[i + tau];
                diff += delta * delta;
            }
            running += diff;
            direct[tau] = diff * tau as f32 / running;
        }

        // FFT-based path under test.
        let mut detector = YinPitchDetector::new();
        detector.cmnd.resize(half_len, 0.0);
        detector.cumulative_mean_normalized_difference(&buffer, half_len);

        let max_abs = (1..half_len)
            .map(|tau| (detector.cmnd[tau] - direct[tau]).abs())
            .fold(0.0f32, f32::max);
        // f32 FFT round-off (difference of large energies) gives ~1e-3 on the
        // O(1) CMND values; a wrong formula or normalization would diverge by
        // orders of magnitude, which this still catches.
        assert!(
            max_abs < 5e-3,
            "FFT CMND diverges from the direct loop: max abs diff {max_abs}"
        );
    }

    #[test]
    fn yin_detects_440hz() {
        let mut detector = YinPitchDetector::new();
        let buffer = generate_sine(440.0, 1024);
        let freq = detector.detect(&buffer).unwrap();
        approx::assert_abs_diff_eq!(freq, 440.0, epsilon = 2.0);
    }

    #[test]
    fn yin_detects_220hz() {
        let mut detector = YinPitchDetector::new();
        let buffer = generate_sine(220.0, 1024);
        let freq = detector.detect(&buffer).unwrap();
        approx::assert_abs_diff_eq!(freq, 220.0, epsilon = 2.0);
    }

    #[test]
    fn yin_detects_100hz() {
        let mut detector = YinPitchDetector::new();
        let buffer = generate_sine(100.0, 2048);
        let freq = detector.detect(&buffer).unwrap();
        approx::assert_abs_diff_eq!(freq, 100.0, epsilon = 2.0);
    }

    #[test]
    fn yin_uses_configured_sample_rate() {
        // A 440Hz tone sampled at 48kHz (the typical web AudioContext rate)
        // is only detected correctly if the detector knows that rate.
        let freq = 440.0_f32;
        let buf: Vec<f32> = (0..BUFFER_SIZE)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / 48_000.0).sin())
            .collect();

        // Rate-aware detector recovers the true pitch.
        let mut det48 = YinPitchDetector::with_sample_rate(48_000.0);
        let detected = det48.detect(&buf).expect("should detect the tone");
        approx::assert_abs_diff_eq!(detected, freq, epsilon = 2.0);

        // The native-default (44.1kHz) detector misreads the same buffer by
        // the rate ratio (~404Hz) — this is the web pitch-skew bug.
        let mut det44 = YinPitchDetector::new();
        let wrong = det44.detect(&buf).expect("still detects a pitch");
        let expected_wrong = freq * 44_100.0 / 48_000.0;
        approx::assert_abs_diff_eq!(wrong, expected_wrong, epsilon = 2.0);
        assert!(
            (detected - wrong).abs() > 30.0,
            "fixed-rate detection should be skewed by >30Hz, got {:.1}Hz vs {:.1}Hz",
            detected,
            wrong
        );
    }

    #[test]
    fn parabolic_interpolation_handles_flat_input() {
        // Collinear/flat neighbours give a zero denominator; result must be
        // finite (fall back to the integer tau) rather than NaN/inf.
        let flat = vec![0.5f32; 8];
        let r = parabolic_interpolation(&flat, 3);
        assert!(r.is_finite(), "flat input produced non-finite tau: {r}");
        let linear = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let r2 = parabolic_interpolation(&linear, 3);
        assert!(r2.is_finite(), "linear input produced non-finite tau: {r2}");
    }

    #[test]
    fn yin_detects_quiet_tone_below_old_fixed_gate() {
        // Amplitude 0.01 => frame energy ~5e-5, below the old fixed 1e-4 gate
        // that would have rejected it. The adaptive gate keeps it.
        let amp = 0.01_f32;
        let freq = 440.0_f32;
        let buf: Vec<f32> = (0..BUFFER_SIZE)
            .map(|i| amp * (std::f32::consts::TAU * freq * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let energy: f32 = buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32;
        assert!(
            energy < 1e-4,
            "test setup: energy {energy} should be below old gate"
        );

        let mut detector = YinPitchDetector::new();
        let detected = detector
            .detect(&buf)
            .expect("quiet clear tone should detect");
        approx::assert_abs_diff_eq!(detected, freq, epsilon = 2.0);
    }

    #[test]
    fn yin_detects_fundamental_of_harmonic_tone() {
        // Sawtooth-like tone: fundamental plus decaying harmonics. YIN must
        // lock onto the fundamental, not an overtone (octave-error guard on a
        // realistic, harmonically-rich signal rather than a pure sine).
        for &f in &[110.0_f32, 196.0, 261.63] {
            let buf: Vec<f32> = (0..BUFFER_SIZE)
                .map(|i| {
                    let t = i as f32 / SAMPLE_RATE as f32;
                    let mut s = 0.0;
                    for k in 1..=6 {
                        s += (1.0 / k as f32) * (std::f32::consts::TAU * f * k as f32 * t).sin();
                    }
                    s * 0.3
                })
                .collect();
            let mut det = YinPitchDetector::new();
            let d = det.detect(&buf).expect("harmonic tone should detect");
            approx::assert_abs_diff_eq!(d, f, epsilon = f * 0.03);
        }
    }

    #[test]
    fn yin_tracks_vibrato() {
        // 5Hz vibrato, +/-1 semitone around A3 (220Hz). Each windowed
        // detection should stay within the vibrato bounds (with a small
        // margin), i.e. no octave jumps or dropouts under modulation.
        use std::f32::consts::TAU;
        let center = 220.0_f32;
        let rate = 5.0;
        let depth = 1.0; // semitones
        let total = BUFFER_SIZE * 20;
        let mut phase = 0.0f32;
        let mut sig = Vec::with_capacity(total);
        for i in 0..total {
            let t = i as f32 / SAMPLE_RATE as f32;
            let f = center * 2f32.powf(depth * (TAU * rate * t).sin() / 12.0);
            phase += f / SAMPLE_RATE as f32;
            phase -= phase.floor();
            sig.push((phase * TAU).sin());
        }
        let lo = center * 2f32.powf(-1.5 / 12.0);
        let hi = center * 2f32.powf(1.5 / 12.0);
        let mut det = YinPitchDetector::new();
        let (mut n, mut ok) = (0u32, 0u32);
        let mut start = 0;
        while start + BUFFER_SIZE <= sig.len() {
            if let Some(f) = det.detect(&sig[start..start + BUFFER_SIZE]) {
                n += 1;
                if f >= lo && f <= hi {
                    ok += 1;
                }
            }
            start += BUFFER_SIZE / 2;
        }
        assert!(n > 0, "no detections across vibrato signal");
        assert!(
            ok as f32 / n as f32 >= 0.8,
            "vibrato tracking only {ok}/{n} within bounds"
        );
    }

    #[test]
    fn yin_returns_none_for_silence() {
        let mut detector = YinPitchDetector::new();
        let buffer = vec![0.0; 1024];
        assert!(detector.detect(&buffer).is_none());
    }

    #[test]
    fn yin_returns_none_for_noise() {
        let mut detector = YinPitchDetector::new();
        let buffer: Vec<f32> = (0..1024)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        let _ = detector.detect(&buffer);
    }

    /// Tests YIN for octave errors across a range of frequencies.
    ///
    /// Octave errors (detecting 2× or 0.5× the true frequency) are YIN's
    /// most common failure mode. Tests pure sines from 60Hz to 2000Hz and
    /// reports any detection that is off by roughly an octave.
    #[test]
    fn perf_yin_octave_error_rate() {
        let mut detector = YinPitchDetector::new();
        let test_freqs: Vec<f32> = [
            60.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 660.0, 880.0,
            1000.0, 1500.0, 2000.0,
        ]
        .into();

        let mut octave_errors = 0;
        let mut tested = 0;

        for &freq in &test_freqs {
            let buffer = generate_sine(freq, BUFFER_SIZE);
            if let Some(detected) = detector.detect(&buffer) {
                tested += 1;
                let ratio = detected / freq;
                let cents = (1200.0 * ratio.log2()).abs();
                let is_octave_error = (cents - 1200.0).abs() < 100.0 // off by ~1 octave
                    || (cents - 2400.0).abs() < 100.0; // off by ~2 octaves
                if is_octave_error {
                    octave_errors += 1;
                }
            }
        }

        eprintln!("[PERF] yin_octave_error_rate: errors={octave_errors}/{tested} (threshold: 0)");
        assert!(
            octave_errors == 0,
            "YIN produced {octave_errors}/{tested} octave errors on pure sines"
        );
    }

    /// Measures fine pitch accuracy in cents across a range of frequencies.
    ///
    /// Reports mean absolute error and worst-case error. Standard metric
    /// in MIR: "Fine Pitch Error" (FPE).
    #[test]
    fn perf_yin_fine_pitch_accuracy() {
        let mut detector = YinPitchDetector::new();
        // Test at many frequencies spanning the vocal range
        let test_freqs: Vec<f32> = (0..30)
            .map(|i| 80.0 * 2.0f32.powf(i as f32 / 6.0)) // 80Hz to ~2560Hz
            .filter(|&f| f <= 2000.0)
            .collect();

        let mut total_cents_error = 0.0f32;
        let mut worst_cents = 0.0f32;
        let mut worst_freq = 0.0f32;
        let mut tested = 0;

        for &freq in &test_freqs {
            let buffer = generate_sine(freq, BUFFER_SIZE);
            if let Some(detected) = detector.detect(&buffer) {
                let cents_error = (1200.0 * (detected / freq).log2()).abs();
                total_cents_error += cents_error;
                tested += 1;
                if cents_error > worst_cents {
                    worst_cents = cents_error;
                    worst_freq = freq;
                }
            }
        }

        let mean_cents = total_cents_error / tested as f32;
        eprintln!(
            "[PERF] yin_fine_pitch_mean_error: {mean_cents:.2} cents (threshold: <{PERF_YIN_MEAN_CENTS})"
        );
        eprintln!(
            "[PERF] yin_fine_pitch_worst_error: {worst_cents:.2} cents at {worst_freq:.1}Hz (threshold: <{PERF_YIN_WORST_CENTS})"
        );

        assert!(
            mean_cents < PERF_YIN_MEAN_CENTS,
            "Mean pitch error {mean_cents:.2} cents exceeds {PERF_YIN_MEAN_CENTS} cent limit"
        );
        assert!(
            worst_cents < PERF_YIN_WORST_CENTS,
            "Worst pitch error {worst_cents:.2} cents at {worst_freq:.1}Hz exceeds {PERF_YIN_WORST_CENTS} cent limit"
        );
    }

    /// Measure what fraction of output energy falls within ±tolerance_bins
    /// of the expected frequency after pitch-shifting a 440Hz sine.
    ///
    /// Uses a 4096-point FFT on the output for ~10.8Hz bin resolution.
    /// Expectation: a clean phase vocoder should concentrate ≥95% of energy
    /// within ±3 bins (~32Hz) of the target frequency. With known bugs
    /// (e.g. off-by-one in expected_phase_advance) this will drop
    /// significantly.
    /// Octave-shift a pure tone and measure the loudest spurious sideband
    /// relative to the shifted fundamental.
    ///
    /// Resample-based pitch shifting stretched each peak's main lobe and
    /// radiated a comb of sidebands at the STFT hop rate (~86 Hz), the worst
    /// only ~13 dB down. Peak-translation shifting with identity phase locking
    /// keeps the lobe narrow and each region phase-coherent, pushing the worst
    /// sideband well below that. This test guards the suppression.
    #[test]
    fn phase_vocoder_octave_shift_sidebands() {
        const ANALYSIS_SIZE: usize = 8192;
        let input_freq = 440.0;
        let ratio = 2.0; // octave up -> a single tone at 880 Hz is ideal
        let mut processor = PhaseVocoderPitchShifter::new(ratio);

        let num_samples = BUFFER_SIZE * 32;
        let mut output = Vec::new();
        for i in 0..num_samples {
            let s = (std::f32::consts::TAU * input_freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(s);
            while let Some(o) = processor.pop_sample() {
                output.push(o);
            }
        }

        let skip = output.len() / 2;
        let block = &output[skip..skip + ANALYSIS_SIZE];
        let window: Vec<f32> = apodize::hanning_iter(ANALYSIS_SIZE)
            .map(|w| w as f32)
            .collect();
        let windowed: Vec<f32> = block.iter().zip(&window).map(|(s, w)| s * w).collect();
        let mags: Vec<f32> = windowed
            .real_fft()
            .get_frequency_bins()
            .iter()
            .map(|b| b.norm())
            .collect();
        let bin_hz = SAMPLE_RATE as f32 / ANALYSIS_SIZE as f32;
        let freq_of = |k: usize| (k + 1) as f32 * bin_hz;

        // Fundamental = global peak (~880 Hz).
        let (main_k, &main_mag) = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let main_freq = freq_of(main_k);

        // Loudest local-max peak outside a guard band around the fundamental.
        const GUARD_HZ: f32 = 40.0;
        let mut worst_db = f32::NEG_INFINITY;
        let mut worst_freq = 0.0f32;
        for k in 1..mags.len() - 1 {
            let f = freq_of(k);
            if f < 100.0 || (f - main_freq).abs() < GUARD_HZ {
                continue;
            }
            if mags[k] > mags[k - 1] && mags[k] >= mags[k + 1] {
                let db = 20.0 * (mags[k] / main_mag).log10();
                if db > worst_db {
                    worst_db = db;
                    worst_freq = f;
                }
            }
        }

        eprintln!(
            "[SIDEBAND] fundamental {main_freq:.1}Hz  worst sideband {worst_freq:.1}Hz at {worst_db:.1} dB"
        );

        // Peak translation should keep the worst sideband well below the
        // fundamental. Baseline (resample shifting) was ~-13 dB; the off-by-one
        // bin fix pushed it to ~-48 dB. Require < -40 dB, leaving ~8 dB for
        // f32/FFT variation across platforms.
        assert!(
            worst_db < -40.0,
            "octave-shift sideband not suppressed: worst {worst_db:.1} dB at {worst_freq:.1}Hz (want < -40 dB)"
        );
    }

    /// Worst-quartile spectral concentration of the fundamental when a
    /// *gliding* pitch is shifted down an octave. Drives an exponential
    /// 100->200 Hz glide through a 0.5x shifter and, per steady-state STFT
    /// block, measures the fraction of output energy within +/-2 bins (~22 Hz)
    /// of that block's peak, then averages the worst quarter of blocks. 1.0 is
    /// a clean fundamental; smearing spreads energy out and lowers it.
    fn downshift_glide_concentration() -> f32 {
        const ANALYSIS: usize = 4096;
        let mut processor = PhaseVocoderPitchShifter::new(0.5);

        // ~1.5 s exponential glide across 100 -> 200 Hz. The fast rate drives
        // strong FM sidebands, so each analysis frame has many low-level peaks.
        let num_samples = BUFFER_SIZE * 32;
        let mut phase = 0.0f32;
        let mut output = Vec::new();
        for i in 0..num_samples {
            let frac = i as f32 / num_samples as f32;
            let freq = 100.0 * 2.0f32.powf(frac);
            phase += freq / SAMPLE_RATE as f32;
            phase -= phase.floor();
            processor.push_sample((phase * std::f32::consts::TAU).sin());
            while let Some(o) = processor.pop_sample() {
                output.push(o);
            }
        }

        let window: Vec<f32> = apodize::hanning_iter(ANALYSIS).map(|w| w as f32).collect();
        let start = output.len() / 4;
        let end = output.len() * 3 / 4;
        let hop = ANALYSIS / 2;
        let mut concs = Vec::new();
        let mut pos = start;
        while pos + ANALYSIS <= end {
            let windowed: Vec<f32> = output[pos..pos + ANALYSIS]
                .iter()
                .zip(&window)
                .map(|(s, w)| s * w)
                .collect();
            let mags: Vec<f32> = windowed
                .real_fft()
                .get_frequency_bins()
                .iter()
                .map(|b| b.norm_sqr())
                .collect();
            let (pk, _) = mags
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let total: f32 = mags.iter().sum();
            let lo = pk.saturating_sub(2);
            let hi = (pk + 2).min(mags.len() - 1);
            let near: f32 = mags[lo..=hi].iter().sum();
            concs.push(near / total);
            pos += hop;
        }
        // Worst-quartile mean: the smear comes in bursts (when the gliding
        // peak crosses bin boundaries), so the worst blocks separate a smeared
        // shift from a clean one far better than the overall mean.
        concs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = (concs.len() / 4).max(1);
        concs[..q].iter().sum::<f32>() / q as f32
    }

    /// A steady tone shifts down an octave cleanly, and now a *gliding* pitch
    /// does too. Previously each analysis frame's low-level leakage /
    /// FM-sideband ripples registered as spectral peaks and were translated
    /// with their own downshift-dependent offset, overwriting the main lobe's
    /// region and jittering the output pitch. Ignoring negligible peaks
    /// (PEAK_REL_THRESHOLD) keeps the gliding fundamental intact.
    #[test]
    fn phase_vocoder_downshift_glide_smears() {
        let conc = downshift_glide_concentration();
        eprintln!("[GLIDE] down-octave glide fundamental concentration: {conc:.4}");
        assert!(
            conc >= 0.97,
            "gliding-pitch fundamental smeared (worst-quartile concentration >= 0.97 expected), got {conc:.4}"
        );
    }

    /// Realized pitch-shift error, in cents, when the shifter is commanded to
    /// move `base_freq` by `cents`. Measures the output fundamental by
    /// autocorrelation with parabolic interpolation on the steady-state tail,
    /// which resolves the period to well under a cent at these frequencies.
    fn shift_error_cents(base_freq: f32, cents: f32) -> f32 {
        let ratio = 2f32.powf(cents / 1200.0);
        let mut processor = PhaseVocoderPitchShifter::new(ratio);
        let num_samples = BUFFER_SIZE * 48;
        let mut output = Vec::new();
        for i in 0..num_samples {
            let s = (std::f32::consts::TAU * base_freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(s);
            while let Some(o) = processor.pop_sample() {
                output.push(o);
            }
        }

        // Steady-state tail, mean-removed.
        let tail = &output[output.len() / 2..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        let x: Vec<f32> = tail.iter().map(|v| v - mean).collect();
        let ac = |lag: usize| -> f32 {
            x[..x.len() - lag]
                .iter()
                .zip(&x[lag..])
                .map(|(a, b)| a * b)
                .sum()
        };

        let expected_period = SAMPLE_RATE as f32 / (base_freq * ratio);
        let lo = (expected_period * 0.8) as usize;
        let hi = (expected_period * 1.25) as usize + 1;
        let mut best = lo;
        let mut best_v = ac(lo);
        for lag in lo..=hi {
            let v = ac(lag);
            if v > best_v {
                best_v = v;
                best = lag;
            }
        }
        let (a, b, c) = (ac(best - 1), ac(best), ac(best + 1));
        let denom = a - 2.0 * b + c;
        let peak = if denom.abs() > 0.0 {
            0.5 * (a - c) / denom
        } else {
            0.0
        };
        let out_freq = SAMPLE_RATE as f32 / (best as f32 + peak);
        let realized = 1200.0 * (out_freq / base_freq).log2();
        (realized - cents).abs()
    }

    /// The phase-vocoder shifter carries pitch in the per-bin phase advance,
    /// not the (coarse) integer bin the magnitude lobe lands on, so it should
    /// resolve very small pitch moves even at low frequency where bins are
    /// ~21 Hz wide. Command a spread of sub-semitone shifts at 100/200 Hz and
    /// require the realized shift to match within a small cent tolerance.
    #[test]
    fn perf_phase_vocoder_fine_shift_resolution() {
        let cases = [
            (100.0, 5.0),
            (100.0, 10.0),
            (100.0, 25.0),
            (100.0, -10.0),
            (100.0, -25.0),
            (200.0, 5.0),
            (200.0, -20.0),
        ];
        let mut worst = 0.0f32;
        for (f, c) in cases {
            let e = shift_error_cents(f, c);
            eprintln!("[PERF]   fine_shift {f:.0}Hz {c:+.0}c -> error {e:.2}c");
            worst = worst.max(e);
        }
        eprintln!(
            "[PERF] phase_vocoder_fine_shift_worst_error: {worst:.2} cents (threshold: <{PERF_FINE_SHIFT_CENTS})"
        );
        assert!(
            worst < PERF_FINE_SHIFT_CENTS,
            "shifter cannot resolve fine pitch moves: worst error {worst:.2} cents (want < {PERF_FINE_SHIFT_CENTS})"
        );
    }

    /// Synthesize a voiced tone the way pitch-tracker benchmarks do: a
    /// harmonic complex (`n_harm` harmonics, 1/k amplitude — a sawtooth-ish
    /// glottal source) with sinusoidal vibrato (`vib_rate_hz`, +/- `vib_cents`)
    /// and additive white noise at `snr_db`. Returns the signal and the
    /// per-sample true f0.
    fn synth_voice(
        f0_hz: f32,
        n_harm: usize,
        vib_rate_hz: f32,
        vib_cents: f32,
        snr_db: f32,
        num: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let sr = SAMPLE_RATE as f32;
        let a_sum: f32 = (1..=n_harm).map(|k| 1.0 / k as f32).sum();
        let mut phase = 0.0f32; // fundamental phase in cycles
        let mut sig = vec![0.0f32; num];
        let mut truth = vec![0.0f32; num];
        for i in 0..num {
            let t = i as f32 / sr;
            let inst = f0_hz
                * 2f32.powf(vib_cents / 1200.0 * (std::f32::consts::TAU * vib_rate_hz * t).sin());
            truth[i] = inst;
            phase += inst / sr;
            let mut s = 0.0;
            for k in 1..=n_harm {
                s += (std::f32::consts::TAU * k as f32 * phase).sin() / k as f32;
            }
            sig[i] = s / a_sum;
        }
        let sig_rms = (sig.iter().map(|v| v * v).sum::<f32>() / num as f32).sqrt();
        let noise_rms = sig_rms * 10f32.powf(-snr_db / 20.0);
        for v in sig.iter_mut() {
            *v += (rng.random::<f32>() * 2.0 - 1.0) * noise_rms * 1.732; // uniform -> matched RMS
        }
        (sig, truth)
    }

    /// Standard pitch-tracker metrics (MIREX-style) over a synthetic voice:
    /// Raw Pitch Accuracy (fraction within 50 cents of the window-mean truth),
    /// Gross Pitch Error fraction (>50 cents off), and Fine Pitch Error (mean
    /// |cents| on the non-gross frames).
    fn yin_voice_metrics(f0_hz: f32, vib_cents: f32, snr_db: f32) -> (f32, f32, f32) {
        let (sig, truth) = synth_voice(
            f0_hz,
            12,
            5.5,
            vib_cents,
            snr_db,
            SAMPLE_RATE * 2,
            0x0F00_D1CE,
        );
        let mut det = YinPitchDetector::new();
        let w = BUFFER_SIZE;
        let hop = 512;
        let (mut total, mut within, mut gross) = (0usize, 0usize, 0usize);
        let mut fine = Vec::new();
        let mut i = 0;
        while i + w <= sig.len() {
            if let Some(f) = det.detect(&sig[i..i + w]) {
                let gt = truth[i..i + w].iter().sum::<f32>() / w as f32;
                let c = 1200.0 * (f / gt).log2();
                total += 1;
                if c.abs() < 50.0 {
                    within += 1;
                    fine.push(c.abs());
                } else {
                    gross += 1;
                }
            }
            i += hop;
        }
        let fpe = fine.iter().sum::<f32>() / fine.len().max(1) as f32;
        (
            within as f32 / total as f32,
            gross as f32 / total as f32,
            fpe,
        )
    }

    #[test]
    fn perf_yin_voice_accuracy() {
        // Realistic sung note: strong harmonics, gentle +/-25 cent 5.5 Hz
        // vibrato, mild noise. This is far harder than a pure tone and is what
        // actually limits the pitch corrector on real singing.
        for (f0, snr) in [(120.0, 40.0), (220.0, 40.0), (120.0, 25.0)] {
            let (rpa, gpe, fpe) = yin_voice_metrics(f0, 25.0, snr);
            eprintln!(
                "[PERF]   voice {f0:.0}Hz SNR{snr:.0}dB: RPA {:.1}%  GPE {:.1}%  FPE {fpe:.1}c",
                rpa * 100.0,
                gpe * 100.0
            );
        }
        let (rpa, _gpe, fpe) = yin_voice_metrics(150.0, 25.0, 40.0);
        eprintln!(
            "[PERF] yin_voice_accuracy: RPA {:.1}% (threshold: >{:.0}%)  FPE {fpe:.1}c (threshold: <{})",
            rpa * 100.0,
            PERF_VOICE_RPA * 100.0,
            PERF_VOICE_FPE_CENTS
        );
        assert!(
            rpa > PERF_VOICE_RPA,
            "voice raw pitch accuracy {:.1}% below {:.0}%",
            rpa * 100.0,
            PERF_VOICE_RPA * 100.0
        );
        assert!(
            fpe < PERF_VOICE_FPE_CENTS,
            "voice fine pitch error {fpe:.1}c exceeds {PERF_VOICE_FPE_CENTS}c"
        );
    }

    fn measure_spectral_purity(scaling_ratio: f32) -> (f32, usize, usize) {
        const ANALYSIS_SIZE: usize = 4096;
        let input_freq = 440.0;
        let expected_freq = input_freq * scaling_ratio;
        let mut processor = PhaseVocoderPitchShifter::new(scaling_ratio);

        // Feed enough signal to reach steady state; skip=len/2 leaves ~8
        // buffers of warmup before the single analysis block.
        let num_samples = BUFFER_SIZE * 16;
        let mut output = Vec::new();
        for i in 0..num_samples {
            let sample = (std::f32::consts::TAU * input_freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
            while let Some(o) = processor.pop_sample() {
                output.push(o);
            }
        }

        // Skip transients, take a steady-state block
        let skip = output.len() / 2;
        assert!(
            output.len() >= skip + ANALYSIS_SIZE,
            "Not enough output: {}",
            output.len()
        );
        let block = &output[skip..skip + ANALYSIS_SIZE];

        // Window before analysis to reduce leakage
        let window: Vec<f32> = apodize::hanning_iter(ANALYSIS_SIZE)
            .map(|w| w as f32)
            .collect();
        let windowed: Vec<f32> = block.iter().zip(&window).map(|(s, w)| s * w).collect();

        let spectrum = windowed.real_fft();
        let bins = spectrum.get_frequency_bins();

        let bin_hz = SAMPLE_RATE as f32 / ANALYSIS_SIZE as f32;

        // Print top 10 bins for diagnostics
        let mut indexed: Vec<(usize, f32)> = bins
            .iter()
            .enumerate()
            .map(|(i, b)| (i, b.norm_sqr()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total_energy: f32 = indexed.iter().map(|(_, e)| e).sum();

        // Find expected bin and measure energy concentration
        let expected_bin = (expected_freq / bin_hz).round() as usize;
        let tolerance_bins = 3;

        let lo = expected_bin.saturating_sub(tolerance_bins);
        let hi = (expected_bin + tolerance_bins).min(bins.len() - 1);

        let band_energy: f32 = bins[lo..=hi].iter().map(|b| b.norm_sqr()).sum();

        let peak_bin = bins
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .unwrap()
            .0;

        let concentration = band_energy / total_energy;
        (concentration, expected_bin, peak_bin)
    }

    #[test]
    fn perf_phase_vocoder_shift_up_fifth_spectral_purity() {
        use crate::music::Interval;
        let ratio = Interval::PERFECT_FIFTH.to_ratio();
        let (concentration, _, _) = measure_spectral_purity(ratio);

        eprintln!(
            "[PERF] phase_vocoder_shift_up_fifth_purity: {:.1}% (threshold: >{:.0}%)",
            concentration * 100.0,
            PERF_SPECTRAL_PURITY * 100.0
        );

        assert!(
            concentration > PERF_SPECTRAL_PURITY,
            "Shift up a fifth: only {:.1}% energy near expected freq (expected ≥{:.0}%).",
            concentration * 100.0,
            PERF_SPECTRAL_PURITY * 100.0,
        );
    }

    #[test]
    fn perf_phase_vocoder_shift_down_fifth_spectral_purity() {
        use crate::music::Interval;
        let ratio = Interval::PERFECT_FIFTH.negate().to_ratio();
        let (concentration, _, _) = measure_spectral_purity(ratio);

        eprintln!(
            "[PERF] phase_vocoder_shift_down_fifth_purity: {:.1}% (threshold: >{:.0}%)",
            concentration * 100.0,
            PERF_SPECTRAL_PURITY * 100.0
        );

        assert!(
            concentration > PERF_SPECTRAL_PURITY,
            "Shift down a fifth: only {:.1}% energy near expected freq (expected ≥{:.0}%).",
            concentration * 100.0,
            PERF_SPECTRAL_PURITY * 100.0,
        );
    }

    #[test]
    fn perf_phase_vocoder_ratio_transition_distortion() {
        use std::sync::atomic::{AtomicU32, Ordering};

        const ANALYSIS_SIZE: usize = 4096;
        let input_freq = 440.0;
        let ratio_before = 1.0f32;
        let ratio_after = 2.0f32.powf(1.0 / 12.0); // half-note up
        let freq_before = input_freq * ratio_before; // 440Hz
        let freq_after = input_freq * ratio_after; // ~466Hz

        let ratio = Arc::new(AtomicU32::new(ratio_before.to_bits()));
        let ratio_clone = ratio.clone();
        let mut processor = PhaseVocoderPitchShifter::with_ratio_fn(move |_: &[f32]| {
            f32::from_bits(ratio_clone.load(Ordering::Relaxed))
        });

        let total_samples = BUFFER_SIZE * 32;
        let switch_at = total_samples / 2;

        let mut output = Vec::new();
        for i in 0..total_samples {
            if i == switch_at {
                ratio.store(ratio_after.to_bits(), Ordering::Relaxed);
            }
            let sample = (std::f32::consts::TAU * input_freq * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
            while let Some(o) = processor.pop_sample() {
                output.push(o);
            }
        }

        // Analyze in sliding windows. For each window, measure energy at
        // the two expected frequencies vs total energy.
        let bin_hz = SAMPLE_RATE as f32 / ANALYSIS_SIZE as f32;
        let bin_before = (freq_before / bin_hz).round() as usize;
        let bin_after = (freq_after / bin_hz).round() as usize;
        let tolerance = 3;
        let window: Vec<f32> = apodize::hanning_iter(ANALYSIS_SIZE)
            .map(|w| w as f32)
            .collect();

        let step = ANALYSIS_SIZE / 4;
        let delay = total_samples - output.len();
        // Output sample index where the ratio switch happens
        let switch_output = switch_at.saturating_sub(delay);

        let mut worst_purity = 1.0f32;
        let mut worst_pos = 0usize;
        let mut transition_windows = 0;
        let mut transition_purity_sum = 0.0f32;

        let mut pos = 0;
        while pos + ANALYSIS_SIZE <= output.len() {
            let windowed: Vec<f32> = output[pos..pos + ANALYSIS_SIZE]
                .iter()
                .zip(&window)
                .map(|(s, w)| s * w)
                .collect();
            let spectrum = windowed.real_fft();
            let bins = spectrum.get_frequency_bins();
            let total: f32 = bins.iter().map(|b| b.norm_sqr()).sum();
            if total < 1e-20 {
                pos += step;
                continue;
            }

            // Energy in the union of both expected bands
            let lo = bin_before.saturating_sub(tolerance);
            let hi = (bin_after + tolerance).min(bins.len() - 1);
            let band_energy: f32 = bins[lo..=hi].iter().map(|b| b.norm_sqr()).sum();
            let purity = band_energy / total;

            // Is this window near the transition?
            let window_center = pos + ANALYSIS_SIZE / 2;
            let near_transition =
                (window_center as i64 - switch_output as i64).unsigned_abs() < ANALYSIS_SIZE as u64;

            if near_transition {
                transition_windows += 1;
                transition_purity_sum += purity;
            }

            if purity < worst_purity {
                worst_purity = purity;
                worst_pos = pos;
            }

            pos += step;
        }

        let transition_avg = if transition_windows > 0 {
            transition_purity_sum / transition_windows as f32
        } else {
            1.0
        };

        eprintln!(
            "[PERF] phase_vocoder_ratio_transition_worst: {:.1}% (threshold: >{:.0}%)",
            worst_purity * 100.0,
            PERF_TRANSITION_WORST * 100.0,
        );
        eprintln!(
            "[PERF] phase_vocoder_ratio_transition_avg: {:.1}% (threshold: >{:.0}%)",
            transition_avg * 100.0,
            PERF_TRANSITION_AVG * 100.0,
        );

        // Steady-state should be very clean
        assert!(
            worst_purity > PERF_TRANSITION_WORST,
            "Worst purity {:.1}% — excessive distortion during ratio change \
             (expected >{:.0}% even at transition). Worst at sample {worst_pos}.",
            worst_purity * 100.0,
            PERF_TRANSITION_WORST * 100.0,
        );

        // Transition region average should still be reasonable
        assert!(
            transition_avg > PERF_TRANSITION_AVG,
            "Transition region average purity {:.1}% — too much distortion \
             around ratio change (expected >{:.0}%).",
            transition_avg * 100.0,
            PERF_TRANSITION_AVG * 100.0,
        );
    }

    #[test]
    fn perf_phase_vocoder_no_alloc_after_warmup() {
        let mut processor = PhaseVocoderPitchShifter::new(0.5);

        // Warmup: let it allocate internal buffers and easyfft thread-local scratch
        let warmup = BUFFER_SIZE * 10;
        for i in 0..warmup {
            let sample = (std::f32::consts::TAU * 440.0 * i as f32 / SAMPLE_RATE as f32).sin();
            processor.push_sample(sample);
        }
        while processor.pop_sample().is_some() {}

        // Steady state: no allocations allowed
        assert_no_alloc::assert_no_alloc(|| {
            for i in 0..BUFFER_SIZE * 2 {
                let sample = (std::f32::consts::TAU * 440.0 * i as f32 / SAMPLE_RATE as f32).sin();
                processor.push_sample(sample);
            }
            while processor.pop_sample().is_some() {}
        });
        eprintln!("[PERF] phase_vocoder_no_alloc: pass (threshold: zero allocations)");
    }

    #[test]
    fn perf_ola_no_alloc_after_warmup() {
        use easyfft::dyn_size::realfft::{DynRealFft, DynRealIfft};

        // Test raw easyfft _using calls
        let buf = vec![0.0f32; BUFFER_SIZE];
        let mut spectrum = buf.real_fft();
        let mut out = vec![0.0f32; BUFFER_SIZE];
        spectrum.real_ifft_using(&mut out);
        buf.real_fft_using(&mut spectrum);
        spectrum.real_ifft_using(&mut out);

        assert_no_alloc::assert_no_alloc(|| {
            buf.real_fft_using(&mut spectrum);
            spectrum.real_ifft_using(&mut out);
        });

        // Now test our converter wrapper with LowPassFilter
        let mut converter =
            TimeToFrequencyDomainBlockProcessorConverter::new(LowPassFilter::new(440));
        let mut buf2 = [0.0f32; BUFFER_SIZE];
        converter.process(&mut buf2);
        converter.process(&mut buf2);
        assert_no_alloc::assert_no_alloc(|| {
            converter.process(&mut buf2);
        });
        eprintln!("[PERF] ola_no_alloc: pass (threshold: zero allocations)");
    }

    #[test]
    fn perf_spectrogram_and_yin_no_alloc_after_warmup() {
        use easyfft::dyn_size::realfft::DynRealFft;

        const SPEC_SIZE: usize = SPECTROGRAM_SIZE;

        let mut spec_scratch = vec![0.0f32; SPEC_SIZE];
        let mut spec_spectrum = spec_scratch.real_fft();
        let contour_scratch: Vec<f32> = (0..BUFFER_SIZE).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut detector = YinPitchDetector::new();

        // Warmup
        for (i, s) in spec_scratch.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin();
        }
        spec_scratch.real_fft_using(&mut spec_spectrum);
        detector.detect(&contour_scratch);

        // Refill spectrogram scratch
        for (i, s) in spec_scratch.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin();
        }

        assert_no_alloc::assert_no_alloc(|| {
            // Spectrogram: in-place FFT + bin read
            spec_scratch.real_fft_using(&mut spec_spectrum);
            let bins = spec_spectrum.get_frequency_bins();
            let _mag = bins[1].norm();

            // Contour: reused YIN detector
            let _pitch = detector.detect(&contour_scratch);

            // Waveform: RMS/peak
            let _peak = contour_scratch
                .iter()
                .map(|s| s.abs())
                .fold(0.0f32, f32::max);
            let _rms = (contour_scratch.iter().map(|s| s * s).sum::<f32>()
                / contour_scratch.len() as f32)
                .sqrt();
        });
        eprintln!("[PERF] spectrogram_and_yin_no_alloc: pass (threshold: zero allocations)");
    }
}
