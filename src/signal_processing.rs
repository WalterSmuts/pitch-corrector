use crate::display::SignalDrawer;
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

pub struct NaivePitchHalver;

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
}

pub struct DisplayProcessor {
    buffer: SegQueue<f32>,
    display_buffer: Mutex<Box<[f32]>>,
    buffer_index: AtomicUsize,
    signal_drawer: SignalDrawer,
}

// TODO: Remove unnecessary memory
pub struct OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    previous_buffer: Mutex<Box<[f32]>>,
    previous_half_buffer: Mutex<Box<[f32]>>,
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

impl DisplayProcessor {
    pub fn new(should_clear_screen: bool) -> Self {
        Self {
            buffer: SegQueue::new(),
            display_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE])),
            buffer_index: AtomicUsize::new(0),
            signal_drawer: SignalDrawer::new(should_clear_screen),
        }
    }
}

impl StreamProcessor for DisplayProcessor {
    fn push_sample(&self, sample: f32) {
        self.buffer.push(sample);
    }

    fn pop_sample(&self) -> Option<f32> {
        let sample = self.buffer.pop()?;
        let mut buffer = self.display_buffer.lock().unwrap();
        buffer[self.buffer_index.load(Ordering::Relaxed)] = sample;
        self.buffer_index.fetch_add(1, Ordering::Relaxed);
        if self.buffer_index.load(Ordering::Relaxed) >= BUFFER_SIZE {
            self.buffer_index.swap(0, Ordering::Relaxed);
            self.signal_drawer.draw_data(&buffer);
        }
        Some(sample)
    }
}

impl BlockProcessor for NaivePitchHalver {
    fn process(&self, buffer: &mut [f32]) {
        let mut temp = [0.0; BUFFER_SIZE / 2].to_vec();
        temp.clone_from_slice(&buffer[0..BUFFER_SIZE / 2]);
        for (index, sample) in buffer.iter_mut().enumerate() {
            *sample = temp[index / 2];
        }
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
        }
    }
}

impl BlockProcessor for FrequencyDomainPitchShifter {
    // TODO: Allow for arbitrary stretching by implementing interpolation on the Complex type
    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum = self.forward_fft.make_output_vec();
        self.forward_fft.process(buffer, &mut spectrum).unwrap();
        let mut spectrum_out = self.forward_fft.make_output_vec();
        for (index, sample) in spectrum_out[0..BUFFER_SIZE / 4].iter_mut().enumerate() {
            *sample = spectrum[index * 2];
        }

        self.inverse_fft.process(&mut spectrum_out, buffer).unwrap();
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
        let previous_buffer = &mut self.previous_buffer.lock().unwrap();

        // Get first block to process (second half of previous and first half on current)
        let mut first = previous_buffer[BUFFER_SIZE / 2..].to_vec();
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
        let previous_half_buffer = &mut self.previous_half_buffer.lock().unwrap();
        // Overlap and add first half of first block previous_half_buffer
        for (first_sample, second_sample) in first[..BUFFER_SIZE / 2]
            .iter_mut()
            .zip(previous_half_buffer.to_vec())
        {
            *first_sample += second_sample;
        }

        // Save current buffer for processing next time
        previous_buffer.copy_from_slice(&temp_input_buffer);

        // Save second part of input buffer windowed
        previous_half_buffer.copy_from_slice(&second[BUFFER_SIZE / 2..]);

        buffer.copy_from_slice(&first);
    }
}

impl<T> OverlapAndAddProcessor<T>
where
    T: BlockProcessor,
{
    pub fn new(block_processor: T) -> Self {
        Self {
            previous_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE])),
            previous_half_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE / 2])),
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
