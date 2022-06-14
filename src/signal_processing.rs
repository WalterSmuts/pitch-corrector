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

pub struct DisplayProcessor {
    buffer: SegQueue<f32>,
    display_buffer: Mutex<Box<[f32]>>,
    buffer_index: AtomicUsize,
    signal_drawer: SignalDrawer,
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
        let mut buffer = self.display_buffer.lock().unwrap();
        buffer[self.buffer_index.load(Ordering::Relaxed)] = sample;
        self.buffer_index.fetch_add(1, Ordering::Relaxed);
        if self.buffer_index.load(Ordering::Relaxed) >= BUFFER_SIZE {
            self.buffer_index.swap(0, Ordering::Relaxed);
            self.signal_drawer.draw_data(&buffer);
        }
    }

    fn pop_sample(&self) -> Option<f32> {
        self.buffer.pop()
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
