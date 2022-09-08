use crate::complex_interpolation::ComplexInterpolate;
use crate::interpolation::Interpolate;
use crate::interpolation::InterpolationMethod;
use crossbeam_queue::SegQueue;
use easyfft::dyn_size::realfft::DynRealDft;
use easyfft::dyn_size::realfft::DynRealFft;
use easyfft::dyn_size::realfft::DynRealIfft;
use easyfft::num_complex::Complex;
use log::info;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

const BUFFER_SIZE: usize = 1024;
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

pub struct DisplayProcessor<const I: usize = BUFFER_SIZE> {
    buffer: SegQueue<f32>,
    display_buffer: Arc<Mutex<[f32; I]>>,
    buffer_index: AtomicUsize,
}

pub struct OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    previous_clean_half_buffer: Mutex<Box<[f32]>>,
    previous_processed_half_buffer: Mutex<Box<[f32]>>,
    block_processor: T,
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
    input_buffer: SegQueue<f32>,
    output_buffer: SegQueue<f32>,
    block_processor: T,
}

pub struct TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    frequency_domain_block_processor: T,
}

impl<T> TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    pub fn new(frequency_domain_block_processor: T) -> Self {
        Self {
            frequency_domain_block_processor,
        }
    }
}

impl<T> BlockProcessor for TimeToFrequencyDomainBlockProcessorConverter<T>
where
    T: FrequencyDomainBlockProcessor,
{
    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum = buffer.real_fft();
        self.frequency_domain_block_processor.process(&mut spectrum);
        buffer.copy_from_slice(&spectrum.real_ifft());
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
        if let Some(sample) = self.first.pop_sample() {
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
            input_buffer: SegQueue::new(),
            output_buffer: SegQueue::new(),
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
        self.input_buffer.push(sample);
        if self.input_buffer.len() > BUFFER_SIZE {
            let mut buffer = [0.0; BUFFER_SIZE];
            for sample in &mut buffer {
                *sample = self.input_buffer.pop().unwrap();
            }
            self.block_processor.process(&mut buffer);
            for sample in buffer {
                self.output_buffer.push(sample);
            }
        }
    }
}

impl<const I: usize> DisplayProcessor<I> {
    pub fn new() -> Self {
        info!("Creating new DisplayProcessor of size {}", I);
        Self {
            buffer: SegQueue::new(),
            display_buffer: Arc::new(Mutex::new([0.0; I])),
            buffer_index: AtomicUsize::new(0),
        }
    }

    pub fn clone_display_buffer(&self) -> Arc<Mutex<[f32; I]>> {
        self.display_buffer.clone()
    }
}

impl<const I: usize> StreamProcessor for DisplayProcessor<I> {
    fn push_sample(&self, sample: f32) {
        self.buffer.push(sample);
    }

    fn pop_sample(&self) -> Option<f32> {
        let sample = self.buffer.pop()?;
        let mut buffer = self.display_buffer.lock().unwrap();
        buffer[self.buffer_index.load(Ordering::Relaxed)] = sample;
        self.buffer_index.fetch_add(1, Ordering::Relaxed);
        if self.buffer_index.load(Ordering::Relaxed) >= I {
            self.buffer_index.swap(0, Ordering::Relaxed);
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
pub(crate) use pipeline;

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

        let mut frequncy_bins = vec![Complex::default(); BUFFER_SIZE / 2];
        let frequncy_bins = if let Some(cutoff_bin) = cutoff_bin {
            for bin in frequncy_bins[cutoff_bin..].iter_mut() {
                *bin = Complex::new(1.0, 0.0);
            }
            frequncy_bins.into_boxed_slice()
        } else {
            frequncy_bins.into_boxed_slice()
        };

        let frequency_response = DynRealDft::new(zeroth_bin, &frequncy_bins, BUFFER_SIZE);

        Self { frequency_response }
    }
}

impl FrequencyDomainBlockProcessor for HighPassFilter {
    fn process(&self, spectrum: &mut DynRealDft<f32>) {
        let processed_dyn_real_dft = spectrum as &DynRealDft<f32> * &self.frequency_response;
        spectrum.clone_from(&processed_dyn_real_dft);
    }
}

impl LowPassFilter {
    pub fn new(cutoff_frequency: usize) -> Self {
        info!("Creating new LowPassFilter");
        let cutoff_bin = get_cutoff_bin(cutoff_frequency);
        let zeroth_bin = if cutoff_bin.is_some() { 0.0 } else { 1.0 };

        let mut frequncy_bins = vec![Complex::default(); BUFFER_SIZE / 2];
        let frequncy_bins = if let Some(cutoff_bin) = cutoff_bin {
            for bin in frequncy_bins[..cutoff_bin].iter_mut() {
                *bin = Complex::new(1.0, 0.0);
            }
            frequncy_bins.into_boxed_slice()
        } else {
            frequncy_bins.into_boxed_slice()
        };

        let frequency_response = DynRealDft::new(zeroth_bin, &frequncy_bins, BUFFER_SIZE);

        Self { frequency_response }
    }
}

impl FrequencyDomainBlockProcessor for LowPassFilter {
    fn process(&self, spectrum: &mut DynRealDft<f32>) {
        let processed_dyn_real_dft = spectrum as &DynRealDft<f32> * &self.frequency_response;
        spectrum.clone_from(&processed_dyn_real_dft);
    }
}

impl FrequencyDomainPitchShifter {
    pub fn new() -> Self {
        info!("Creating new FrequencyDomainPitchShifter");
        Self { scaling_ratio: 0.5 }
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

impl<T> BlockProcessor for OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    // TODO: Remove unnecessary allocations
    fn process(&self, buffer: &mut [f32]) {
        // Create a temp clone of input buffer
        let temp_input_buffer = buffer.to_vec();

        // Get a lock and reference to previous buffer
        let previous_clean_half_buffer = &mut self.previous_clean_half_buffer.lock().unwrap();

        // Get first block to process (second half of previous and first half on current)
        let mut first = previous_clean_half_buffer.to_vec();
        first.append(&mut buffer[..BUFFER_SIZE / 2].to_vec());

        // Get second block to process (the current input buffer)
        let mut second = buffer.to_vec();

        // Apply hanning window to first block
        let window = apodize::hanning_iter(BUFFER_SIZE);
        for (sample, window_sample) in first.iter_mut().zip(window) {
            *sample *= window_sample as f32;
        }

        // Apply hanning window to second block
        let window = apodize::hanning_iter(BUFFER_SIZE);
        for (sample, window_sample) in second.iter_mut().zip(window) {
            *sample *= window_sample as f32;
        }

        // Process each block separately
        self.block_processor.process(&mut first);
        self.block_processor.process(&mut second);

        // Overlap and add second half of first block and first half of second block
        for (first_sample, second_sample) in first[BUFFER_SIZE / 2..]
            .iter_mut()
            .zip(&second[..BUFFER_SIZE / 2])
        {
            *first_sample += second_sample;
        }

        // Lock and get reference to previous half buffer
        let previous_processed_half_buffer =
            &mut self.previous_processed_half_buffer.lock().unwrap();

        // Overlap and add first half of first block previous_processed_half_buffer
        for (first_sample, second_sample) in first[..BUFFER_SIZE / 2]
            .iter_mut()
            .zip(previous_processed_half_buffer.to_vec())
        {
            *first_sample += second_sample;
        }

        // Save half current buffer for processing next time
        previous_clean_half_buffer.copy_from_slice(&temp_input_buffer[BUFFER_SIZE / 2..]);

        // Save second part of input buffer windowed and processed
        previous_processed_half_buffer.copy_from_slice(&second[BUFFER_SIZE / 2..]);

        buffer.copy_from_slice(&first);
    }
}

impl<T> OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    #[allow(dead_code)]
    pub fn new(block_processor: T) -> Self {
        info!("Creating new OverlapAndAddProcessor");
        Self {
            previous_clean_half_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE / 2])),
            previous_processed_half_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE / 2])),
            block_processor,
        }
    }
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
        let queue = SegQueue::new();
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>();
            passthrough_stream_processor.push_sample(x);
            queue.push(x);
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
        let queue = SegQueue::new();
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>();
            passthrough_stream_processor.push_sample(x);
            queue.push(x);
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
        let queue = SegQueue::new();
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>();
            passthrough_stream_processor.push_sample(x);
            queue.push(x);
        }

        while let Some(stream_sample) = passthrough_stream_processor.pop_sample() {
            assert_eq!(stream_sample, queue.pop().unwrap());
        }
    }
}
