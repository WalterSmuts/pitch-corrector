use crate::complex_interpolation::ComplexInterpolate;
use crate::display::SignalDrawer;
use crate::interpolation::Interpolate;
use crate::interpolation::InterpolationMethod;
use crossbeam_queue::SegQueue;
use realfft::num_complex::Complex;
use realfft::RealToComplex;
use realfft::{ComplexToReal, RealFftPlanner};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

const BUFFER_SIZE: usize = 1024;

pub trait StreamProcessor {
    fn push_sample(&self, sample: f32);
    fn pop_sample(&self) -> Option<f32>;
}

pub trait BlockProcessor {
    fn process(&self, buffer: &mut [f32]);
}

pub struct NaivePitchShifter {
    scaling_ratio: f32,
}

pub struct HighPassFilter {
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

pub struct LowPassFilter {
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

pub struct FrequencyDomainPitchShifter {
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
    scaling_ratio: f32,
}

pub struct DisplayProcessor<const I: usize = BUFFER_SIZE> {
    buffer: SegQueue<f32>,
    display_buffer: Mutex<[f32; I]>,
    buffer_index: AtomicUsize,
    signal_drawer: SignalDrawer,
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
        Self { first, second }
    }
}

impl<T> Segmenter<T>
where
    T: BlockProcessor,
{
    pub fn new(block_processor: T) -> Self {
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
    pub fn new(should_clear_screen: bool) -> Self {
        Self {
            buffer: SegQueue::new(),
            display_buffer: Mutex::new([0.0; I]),
            buffer_index: AtomicUsize::new(0),
            signal_drawer: SignalDrawer::new(should_clear_screen),
        }
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
            self.signal_drawer.draw_data(&*buffer);
        }
        Some(sample)
    }
}

impl NaivePitchShifter {
    pub fn new(scaling_ratio: f32) -> Self {
        Self { scaling_ratio }
    }
}

impl BlockProcessor for NaivePitchShifter {
    fn process(&self, buffer: &mut [f32]) {
        let mut output_buffer = [0.0; BUFFER_SIZE];
        for (index, sample) in output_buffer.iter_mut().enumerate() {
            *sample = (index as f32 * self.scaling_ratio) % (BUFFER_SIZE as f32 - 1.0);
        }
        Interpolate::<{ InterpolationMethod::Linear }>::interpolate_samples(
            buffer,
            &mut output_buffer,
        );
        buffer.copy_from_slice(&output_buffer);
    }
}

impl HighPassFilter {
    pub fn new() -> Self {
        let mut real_planner = RealFftPlanner::new();
        Self {
            forward_fft: real_planner.plan_fft_forward(BUFFER_SIZE),
            inverse_fft: real_planner.plan_fft_inverse(BUFFER_SIZE),
        }
    }
}

impl BlockProcessor for HighPassFilter {
    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum = self.forward_fft.make_output_vec();
        self.forward_fft.process(buffer, &mut spectrum).unwrap();
        spectrum[0..15]
            .iter_mut()
            .for_each(|sample| *sample = Complex::new(0.0, 0.0));
        self.inverse_fft.process(&mut spectrum, buffer).unwrap();
        for sample in buffer {
            *sample /= BUFFER_SIZE as f32;
        }
    }
}

impl LowPassFilter {
    pub fn new() -> Self {
        let mut real_planner = RealFftPlanner::new();
        Self {
            forward_fft: real_planner.plan_fft_forward(BUFFER_SIZE),
            inverse_fft: real_planner.plan_fft_inverse(BUFFER_SIZE),
        }
    }
}

impl BlockProcessor for LowPassFilter {
    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum = self.forward_fft.make_output_vec();
        self.forward_fft.process(buffer, &mut spectrum).unwrap();
        spectrum[15..]
            .iter_mut()
            .for_each(|sample| *sample = Complex::new(0.0, 0.0));
        self.inverse_fft.process(&mut spectrum, buffer).unwrap();
        for sample in buffer {
            *sample /= BUFFER_SIZE as f32;
        }
    }
}

impl FrequencyDomainPitchShifter {
    pub fn new() -> Self {
        let mut real_planner = RealFftPlanner::new();
        Self {
            forward_fft: real_planner.plan_fft_forward(BUFFER_SIZE),
            inverse_fft: real_planner.plan_fft_inverse(BUFFER_SIZE),
            scaling_ratio: 0.5,
        }
    }
}

impl BlockProcessor for FrequencyDomainPitchShifter {
    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum = self.forward_fft.make_output_vec();
        self.forward_fft.process(buffer, &mut spectrum).unwrap();
        let mut spectrum_out = self.forward_fft.make_output_vec();
        for (index, sample) in spectrum_out[0..BUFFER_SIZE / 2 + 1].iter_mut().enumerate() {
            let index = index as f32 / self.scaling_ratio;
            *sample = if index.ceil() >= spectrum.len() as f32 {
                Complex::default()
            } else {
                spectrum.interpolate_sample(index)
            };
        }

        // Ignore result because we're working with f32's and they may have small values when
        // zero's are expected.
        let _ = self.inverse_fft.process(&mut spectrum_out, buffer);
        for sample in buffer {
            *sample /= BUFFER_SIZE as f32;
        }
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
    pub fn new(block_processor: T) -> Self {
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
