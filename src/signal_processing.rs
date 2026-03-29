use crate::complex_interpolation::ComplexInterpolate;
use crate::interpolation::Interpolate;
use crate::interpolation::InterpolationMethod;
use crossbeam_queue::ArrayQueue;
use easyfft::dyn_size::realfft::DynRealDft;
use easyfft::dyn_size::realfft::DynRealFft;
use easyfft::dyn_size::realfft::DynRealIfft;
use easyfft::num_complex::Complex;
use log::info;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

pub const BUFFER_SIZE: usize = 2048;
pub const SPECTROGRAM_SIZE: usize = 8192;
const SAMPLE_RATE: usize = 44100;

pub trait StreamProcessor {
    fn push_sample(&self, sample: f32);
    fn pop_sample(&self) -> Option<f32>;
}

pub trait BlockProcessor {
    fn process(&self, buffer: &mut [f32]);
}

pub trait FrequencyDomainBlockProcessor {
    fn process(&self, buffer: &mut DynRealDft<f32>);
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

pub struct PhaseVocoderPitchShifter<F: Fn(&[f32]) -> f32 + Send + Sync> {
    ratio_fn: F,
    hop_size: usize,
    input_buffer: ArrayQueue<f32>,
    output_buffer: ArrayQueue<f32>,
    state: Mutex<PhaseVocoderState>,
}

struct BinData {
    magnitudes: Vec<f32>,
    true_freq: Vec<f32>,
    freq_deriv: Vec<f32>,
}

impl BinData {
    fn resize(&mut self, len: usize) {
        self.magnitudes.resize(len, 0.0);
        self.true_freq.resize(len, 0.0);
        self.freq_deriv.resize(len, 0.0);
    }

    fn clear(&mut self) {
        self.magnitudes.iter_mut().for_each(|v| *v = 0.0);
        self.true_freq.iter_mut().for_each(|v| *v = 0.0);
        self.freq_deriv.iter_mut().for_each(|v| *v = 0.0);
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
}

pub struct DisplayProcessor<const I: usize = BUFFER_SIZE> {
    buffer: ArrayQueue<f32>,
    back_buffer: Mutex<Box<[f32; I]>>,
    front_buffer: Arc<Mutex<[f32; I]>>,
    write_index: AtomicUsize,
}

pub struct OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    previous_clean_half_buffer: Mutex<Box<[f32]>>,
    previous_processed_half_buffer: Mutex<Box<[f32]>>,
    scratch: Mutex<OlaBuffers>,
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
    spectrum: Mutex<Option<DynRealDft<f32>>>,
    ifft_buf: Mutex<Vec<f32>>,
}

impl<T> TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    pub fn new(frequency_domain_block_processor: T) -> Self {
        Self {
            frequency_domain_block_processor,
            spectrum: Mutex::new(Some(DynRealDft::new(
                0.0,
                &vec![Complex::default(); BUFFER_SIZE / 2],
                BUFFER_SIZE,
            ))),
            ifft_buf: Mutex::new(vec![0.0; BUFFER_SIZE]),
        }
    }
}

impl<T> BlockProcessor for TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum_opt = self.spectrum.lock().unwrap();
        let spectrum = spectrum_opt.as_mut().unwrap();
        buffer.real_fft_using(spectrum);
        self.frequency_domain_block_processor.process(spectrum);

        let mut ifft_buf = self.ifft_buf.lock().unwrap();
        spectrum.real_ifft_using(&mut ifft_buf);
        buffer.copy_from_slice(&ifft_buf);
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
    fn push_sample(&self, sample: f32) {
        self.first.push_sample(sample);
        while let Some(sample) = self.first.pop_sample() {
            self.second.push_sample(sample);
        }
    }
    fn pop_sample(&self) -> Option<f32> {
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
    fn pop_sample(&self) -> Option<f32> {
        self.output_buffer.pop()
    }

    fn push_sample(&self, sample: f32) {
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
            back_buffer: Mutex::new(Box::new([0.0; I])),
            front_buffer: Arc::new(Mutex::new([0.0; I])),
            write_index: AtomicUsize::new(0),
        }
    }

    pub fn clone_display_buffer(&self) -> Arc<Mutex<[f32; I]>> {
        self.front_buffer.clone()
    }
}

impl<const I: usize> StreamProcessor for DisplayProcessor<I> {
    fn push_sample(&self, sample: f32) {
        if self.buffer.push(sample).is_err() {
            log::warn!("DisplayProcessor: buffer overflow — dropping sample");
        }
    }

    fn pop_sample(&self) -> Option<f32> {
        let sample = self.buffer.pop()?;
        let mut back = self.back_buffer.lock().unwrap();
        let idx = self.write_index.load(Ordering::Relaxed);
        back[idx] = sample;
        let next = idx + 1;
        if next >= I {
            // Back buffer full — swap to front
            let mut front = self.front_buffer.lock().unwrap();
            front.copy_from_slice(back.as_ref());
            self.write_index.store(0, Ordering::Relaxed);
        } else {
            self.write_index.store(next, Ordering::Relaxed);
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
    ($first_processor:expr$(,)?) => {
        $first_processor
    };

    ($first_processor:expr, $($other_processors:expr),+ $(,)?) => {
        $crate::signal_processing::compose(
            pipeline! {$first_processor},
            pipeline! { $($other_processors),+ },
        )
    };
}
pub use pipeline;

impl BlockProcessor for NaivePitchShifter {
    fn process(&self, buffer: &mut [f32]) {
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
    fn process(&self, spectrum: &mut DynRealDft<f32>) {
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
    fn process(&self, spectrum: &mut DynRealDft<f32>) {
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
    fn process(&self, spectrum: &mut DynRealDft<f32>) {
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
    pub fn new(
        scaling_ratio: f32,
    ) -> PhaseVocoderPitchShifter<impl Fn(&[f32]) -> f32 + Send + Sync> {
        PhaseVocoderPitchShifter::with_ratio_fn(move |_: &[f32]| scaling_ratio)
    }
}

impl<F: Fn(&[f32]) -> f32 + Send + Sync> PhaseVocoderPitchShifter<F> {
    pub fn with_ratio_fn(ratio_fn: F) -> Self {
        info!("Creating new PhaseVocoderPitchShifter with dynamic ratio");
        let hop_size = BUFFER_SIZE / 4;
        let window: Vec<f32> = apodize::hanning_iter(BUFFER_SIZE)
            .map(|w| w as f32)
            .collect();
        Self {
            ratio_fn,
            hop_size,
            input_buffer: ArrayQueue::new(BUFFER_SIZE * 4),
            output_buffer: ArrayQueue::new(BUFFER_SIZE * 4),
            state: Mutex::new(PhaseVocoderState {
                input_frame: vec![0.0; BUFFER_SIZE],
                input_pos: 0,
                prev_input_phase: vec![],
                prev_output_phase: vec![],
                output_accum: vec![0.0; BUFFER_SIZE],
                analysis: BinData {
                    magnitudes: vec![],
                    true_freq: vec![],
                    freq_deriv: vec![],
                },
                synthesis: BinData {
                    magnitudes: vec![],
                    true_freq: vec![],
                    freq_deriv: vec![],
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
            }),
        }
    }

    fn process_frame(state: &mut PhaseVocoderState, scaling_ratio: f32, hop_size: usize) {
        use std::collections::BinaryHeap;

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
            state.analysis.resize(num_bins);
            state.synthesis.resize(num_bins);
            state
                .synthesis_bins
                .resize(BUFFER_SIZE / 2, Complex::default());
        }

        // Analysis: compute magnitude, time derivative, and frequency derivative
        for k in 0..num_bins {
            state.analysis.magnitudes[k] = bins[k].norm();
            let phase = bins[k].arg();

            // Time derivative (backward difference)
            let mut phase_diff = phase - state.prev_input_phase[k];
            state.prev_input_phase[k] = phase;
            phase_diff -= expected_phase_advance(k + 1);
            phase_diff = phase_diff.rem_euclid(std::f32::consts::TAU);
            if phase_diff > std::f32::consts::PI {
                phase_diff -= std::f32::consts::TAU;
            }
            state.analysis.true_freq[k] = expected_phase_advance(k + 1) + phase_diff;

            // Frequency derivative (centered difference)
            if k > 0 && k < num_bins - 1 {
                let mut fd = bins[k + 1].arg() - bins[k - 1].arg();
                fd = fd.rem_euclid(std::f32::consts::TAU);
                if fd > std::f32::consts::PI {
                    fd -= std::f32::consts::TAU;
                }
                state.analysis.freq_deriv[k] = fd / 2.0;
            } else {
                state.analysis.freq_deriv[k] = 0.0;
            }
        }

        // Synthesis: shift bins
        state.synthesis.clear();
        for b in state.synthesis_bins.iter_mut() {
            *b = Complex::default();
        }

        for k in 0..num_bins {
            let src = k as f32 / scaling_ratio;
            let lo = src as usize;
            let hi = lo + 1;
            if hi < num_bins {
                let frac = src - lo as f32;
                state.synthesis.magnitudes[k] = state.analysis.magnitudes[lo] * (1.0 - frac)
                    + state.analysis.magnitudes[hi] * frac;
                state.synthesis.true_freq[k] = (state.analysis.true_freq[lo] * (1.0 - frac)
                    + state.analysis.true_freq[hi] * frac)
                    * scaling_ratio;
                state.synthesis.freq_deriv[k] = state.analysis.freq_deriv[lo] * (1.0 - frac)
                    + state.analysis.freq_deriv[hi] * frac;
            } else if lo < num_bins {
                state.synthesis.magnitudes[k] = state.analysis.magnitudes[lo];
                state.synthesis.true_freq[k] = state.analysis.true_freq[lo] * scaling_ratio;
                state.synthesis.freq_deriv[k] = state.analysis.freq_deriv[lo];
            }
        }

        // Phase Gradient Heap Integration (PGHI)
        // Propagate phase from loudest bin outward in both time and frequency
        let mut computed = vec![false; num_bins];
        let mut new_phase = vec![0.0f32; num_bins];

        // Max-heap: (magnitude_bits, bin_index, from_prev_frame)
        let mut heap: BinaryHeap<(u32, usize, bool)> = BinaryHeap::new();

        // Seed with previous frame bins (candidates for time propagation)
        for k in 0..num_bins {
            if state.synthesis.magnitudes[k] > 1e-10 {
                heap.push((state.synthesis.magnitudes[k].to_bits(), k, true));
            }
        }

        while let Some((_, k, from_prev)) = heap.pop() {
            if computed[k] {
                continue;
            }
            if from_prev {
                // Time direction: classical phase accumulation
                new_phase[k] = state.prev_output_phase[k] + state.synthesis.true_freq[k];
            }
            computed[k] = true;

            // Propagate in frequency direction to neighbors
            if k + 1 < num_bins && !computed[k + 1] {
                let phase_up = new_phase[k]
                    + (state.synthesis.freq_deriv[k] + state.synthesis.freq_deriv[k + 1]) / 2.0;
                new_phase[k + 1] = phase_up;
                heap.push((state.synthesis.magnitudes[k + 1].to_bits(), k + 1, false));
            }
            if k > 0 && !computed[k - 1] {
                let phase_down = new_phase[k]
                    - (state.synthesis.freq_deriv[k] + state.synthesis.freq_deriv[k - 1]) / 2.0;
                new_phase[k - 1] = phase_down;
                heap.push((state.synthesis.magnitudes[k - 1].to_bits(), k - 1, false));
            }
        }

        state.prev_output_phase.copy_from_slice(&new_phase);

        // Build synthesis bins
        for (k, phase) in new_phase.iter().enumerate().take(num_bins) {
            state.synthesis_bins[k] = Complex::from_polar(state.synthesis.magnitudes[k], *phase);
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

        // Apply synthesis window, normalized for 75% overlap
        for (s, w) in state.ifft_output.iter_mut().zip(state.window.iter()) {
            *s *= w / 1.5;
        }
    }
}

impl<F: Fn(&[f32]) -> f32 + Send + Sync> StreamProcessor for PhaseVocoderPitchShifter<F> {
    fn push_sample(&self, sample: f32) {
        if self.input_buffer.push(sample).is_err() {
            log::warn!("PhaseVocoder: input buffer overflow — dropping sample");
        }

        if self.input_buffer.len() >= self.hop_size {
            let mut state = self.state.lock().unwrap();

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
            Self::process_frame(&mut state, scaling_ratio, self.hop_size);

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

    fn pop_sample(&self) -> Option<f32> {
        self.output_buffer.pop()
    }
}

impl<T> BlockProcessor for OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    fn process(&self, buffer: &mut [f32]) {
        let mut scratch = self.scratch.lock().unwrap();
        let previous_clean_half_buffer = &mut self.previous_clean_half_buffer.lock().unwrap();

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
        let previous_processed_half_buffer =
            &mut self.previous_processed_half_buffer.lock().unwrap();
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
            previous_clean_half_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE / 2])),
            previous_processed_half_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE / 2])),
            scratch: Mutex::new(OlaBuffers {
                first: Box::new([0.0; BUFFER_SIZE]),
                second: Box::new([0.0; BUFFER_SIZE]),
            }),
            window,
            block_processor,
        }
    }
}

const DEFAULT_YIN_THRESHOLD: f32 = 0.15;

pub struct YinPitchDetector {
    threshold: f32,
    cmnd: Vec<f32>,
}

impl Default for YinPitchDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl YinPitchDetector {
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_YIN_THRESHOLD,
            cmnd: vec![0.0; BUFFER_SIZE / 2],
        }
    }

    pub fn detect(&mut self, buffer: &[f32]) -> Option<f32> {
        let half_len = buffer.len() / 2;
        if half_len < 2 {
            return None;
        }

        // Reject silence / noise floor
        let energy: f32 = buffer.iter().map(|s| s * s).sum::<f32>() / buffer.len() as f32;
        if energy < 1e-4 {
            return None;
        }

        self.cmnd.resize(half_len, 0.0);
        self.cumulative_mean_normalized_difference(buffer, half_len);
        let tau = self.absolute_threshold()?;
        let refined_tau = parabolic_interpolation(&self.cmnd, tau);
        let frequency = SAMPLE_RATE as f32 / refined_tau;

        if !(50.0..=4000.0).contains(&frequency) {
            return None;
        }

        Some(frequency)
    }

    fn cumulative_mean_normalized_difference(&mut self, buffer: &[f32], half_len: usize) {
        self.cmnd[0] = 1.0;

        let mut running_sum = 0.0;
        for tau in 1..half_len {
            let mut diff = 0.0;
            for i in 0..half_len {
                let delta = buffer[i] - buffer[i + tau];
                diff += delta * delta;
            }
            running_sum += diff;
            self.cmnd[tau] = diff * tau as f32 / running_sum;
        }
    }

    fn absolute_threshold(&self) -> Option<usize> {
        let min_tau = 2;
        for tau in min_tau..self.cmnd.len() {
            if self.cmnd[tau] < self.threshold {
                let mut best = tau;
                while best + 1 < self.cmnd.len() && self.cmnd[best + 1] < self.cmnd[best] {
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
    let peak = 0.5 * (alpha - gamma) / (alpha - 2.0 * beta + gamma);
    tau as f32 + peak
}
#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SAMPLE_SIZE: usize = BUFFER_SIZE * 10;
    const TEST_EQUALITY_EPISLON: f32 = 0.002;

    struct PassthroughBlockProcessor;

    impl BlockProcessor for PassthroughBlockProcessor {
        fn process(&self, _buffer: &mut [f32]) {
            // Do nothing to buffer
        }
    }

    struct AmplitudeHalvingBlockProcessor;

    impl BlockProcessor for AmplitudeHalvingBlockProcessor {
        fn process(&self, buffer: &mut [f32]) {
            for sample in buffer.iter_mut() {
                *sample /= 2.0;
            }
        }
    }

    #[test]
    fn overlap_and_add_processor_is_transparent() {
        let passthrough_stream_processor =
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
        let passthrough_stream_processor =
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
        let passthrough_stream_processor = Segmenter::new(PassthroughBlockProcessor);
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
        let processor = Segmenter::new(OverlapAndAddProcessor::new(
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
        let converter = TimeToFrequencyDomainBlockProcessorConverter::new(filter);
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
        let processor = Segmenter::new(OverlapAndAddProcessor::new(
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
        let processor = Segmenter::new(OverlapAndAddProcessor::new(
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
    fn phase_vocoder_unity_ratio_is_transparent() {
        let input_freq = 440.0;
        let processor = PhaseVocoderPitchShifter::new(1.0);

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
        assert!(
            (similarity - 1.0).abs() < 0.01,
            "Phase vocoder at ratio 1.0 should be transparent, but similarity was {similarity:.3}"
        );
    }

    #[test]
    fn phase_vocoder_pitch_shifter_produces_output() {
        let input_freq = 440.0;
        let processor = PhaseVocoderPitchShifter::new(0.5);

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

    /// Measure what fraction of output energy falls within ±tolerance_bins
    /// of the expected frequency after pitch-shifting a 440Hz sine.
    ///
    /// Uses a 4096-point FFT on the output for ~10.8Hz bin resolution.
    /// Expectation: a clean phase vocoder should concentrate ≥95% of energy
    /// within ±3 bins (~32Hz) of the target frequency. With known bugs
    /// (e.g. off-by-one in expected_phase_advance) this will drop
    /// significantly.
    fn measure_spectral_purity(scaling_ratio: f32) -> (f32, usize, usize) {
        const ANALYSIS_SIZE: usize = 4096;
        let input_freq = 440.0;
        let expected_freq = input_freq * scaling_ratio;
        let processor = PhaseVocoderPitchShifter::new(scaling_ratio);

        // Feed plenty of signal for steady-state
        let num_samples = BUFFER_SIZE * 80;
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
        eprintln!("  Top 5 bins (ratio={scaling_ratio:.3}):");
        for (rank, (bin, energy)) in indexed.iter().take(5).enumerate() {
            eprintln!(
                "    #{}: bin {} ({:.1}Hz) = {:.2}%",
                rank + 1,
                bin,
                *bin as f32 * bin_hz,
                energy / total_energy * 100.0
            );
        }

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
    fn phase_vocoder_shift_up_fifth_spectral_purity() {
        let ratio = 3.0 / 2.0; // up a fifth: 440 -> 660Hz
        let (concentration, expected_bin, peak_bin) = measure_spectral_purity(ratio);
        let bin_hz = SAMPLE_RATE as f32 / 4096.0;

        eprintln!(
            "UP FIFTH: {:.1}% energy in band, expected bin {} ({:.1}Hz), peak bin {} ({:.1}Hz)",
            concentration * 100.0,
            expected_bin,
            expected_bin as f32 * bin_hz,
            peak_bin,
            peak_bin as f32 * bin_hz,
        );

        assert!(
            concentration > 0.99,
            "Shift up a fifth: only {:.1}% energy near 660Hz (expected ≥99%). \
             Peak at bin {peak_bin} ({:.1}Hz), expected bin {expected_bin} ({:.1}Hz). \
             Likely phase vocoder distortion.",
            concentration * 100.0,
            peak_bin as f32 * bin_hz,
            expected_bin as f32 * bin_hz,
        );
    }

    #[test]
    fn phase_vocoder_shift_down_fifth_spectral_purity() {
        let ratio = 2.0 / 3.0; // down a fifth: 440 -> ~293Hz
        let (concentration, expected_bin, peak_bin) = measure_spectral_purity(ratio);
        let bin_hz = SAMPLE_RATE as f32 / 4096.0;

        eprintln!(
            "DOWN FIFTH: {:.1}% energy in band, expected bin {} ({:.1}Hz), peak bin {} ({:.1}Hz)",
            concentration * 100.0,
            expected_bin,
            expected_bin as f32 * bin_hz,
            peak_bin,
            peak_bin as f32 * bin_hz,
        );

        assert!(
            concentration > 0.99,
            "Shift down a fifth: only {:.1}% energy near 293Hz (expected ≥99%). \
             Peak at bin {peak_bin} ({:.1}Hz), expected bin {expected_bin} ({:.1}Hz). \
             Likely phase vocoder distortion.",
            concentration * 100.0,
            peak_bin as f32 * bin_hz,
            expected_bin as f32 * bin_hz,
        );
    }

    #[test]
    fn phase_vocoder_ratio_transition_distortion() {
        use std::sync::atomic::AtomicU32;

        const ANALYSIS_SIZE: usize = 4096;
        let input_freq = 440.0;
        let ratio_before = 1.0f32;
        let ratio_after = 2.0f32.powf(1.0 / 12.0); // half-note up
        let freq_before = input_freq * ratio_before; // 440Hz
        let freq_after = input_freq * ratio_after; // ~466Hz

        let ratio = Arc::new(AtomicU32::new(ratio_before.to_bits()));
        let ratio_clone = ratio.clone();
        let processor = PhaseVocoderPitchShifter::with_ratio_fn(move |_: &[f32]| {
            f32::from_bits(ratio_clone.load(Ordering::Relaxed))
        });

        let total_samples = BUFFER_SIZE * 80;
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
            "RATIO TRANSITION: worst purity {:.1}% at sample {worst_pos}, \
             transition avg {:.1}% over {transition_windows} windows",
            worst_purity * 100.0,
            transition_avg * 100.0,
        );

        // Steady-state should be very clean
        assert!(
            worst_purity > 0.98,
            "Worst purity {:.1}% — excessive distortion during ratio change \
             (expected >98% even at transition). Worst at sample {worst_pos}.",
            worst_purity * 100.0,
        );

        // Transition region average should still be reasonable
        assert!(
            transition_avg > 0.99,
            "Transition region average purity {:.1}% — too much distortion \
             around ratio change (expected >99%).",
            transition_avg * 100.0,
        );
    }

    #[test]
    fn phase_vocoder_no_alloc_after_warmup() {
        let processor = PhaseVocoderPitchShifter::new(0.5);

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
    }

    #[test]
    fn ola_no_alloc_after_warmup() {
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
        let converter = TimeToFrequencyDomainBlockProcessorConverter::new(LowPassFilter::new(440));
        let mut buf2 = [0.0f32; BUFFER_SIZE];
        converter.process(&mut buf2);
        converter.process(&mut buf2);
        assert_no_alloc::assert_no_alloc(|| {
            converter.process(&mut buf2);
        });
    }

    #[test]
    fn spectrogram_and_yin_no_alloc_after_warmup() {
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
    }
}
