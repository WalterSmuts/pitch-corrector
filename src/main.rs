use clap::Parser;
use crossbeam_queue::SegQueue;
use display::SignalDrawer;
use realfft::num_complex::Complex;
use realfft::RealToComplex;
use realfft::{ComplexToReal, RealFftPlanner};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

mod display;
mod hardware;

const BUFFER_SIZE: usize = 1024;

#[derive(Parser)]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    Passthrough,
    /// Passthrough microphone to speakers but halve the pitch
    SimplePitchHalver,
    /// Passthrough microphone to speakers but filter out low frequencies
    HighPassFilter,
}
fn passthrough() {
    let _streams = hardware::setup_passthrough_processor(DisplayProcessor::new(true));
    std::thread::park();
}

struct PitchHalver {
    input_buffer: SegQueue<f32>,
    output_buffer: SegQueue<f32>,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

struct HighPassFilter {
    input_buffer: SegQueue<f32>,
    output_buffer: SegQueue<f32>,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

struct ComposedProcessor<F, S>
where
    F: StreamProcessor,
    S: StreamProcessor,
{
    first: F,
    second: S,
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
    fn new(first: F, second: S) -> Self {
        Self { first, second }
    }
}

struct DisplayProcessor {
    buffer: SegQueue<f32>,
    display_buffer: Mutex<Box<[f32]>>,
    buffer_index: AtomicUsize,
    signal_drawer: SignalDrawer,
}

pub trait StreamProcessor {
    fn push_sample(&self, sample: f32);
    fn pop_sample(&self) -> Option<f32>;
}

impl DisplayProcessor {
    fn new(should_clear_screen: bool) -> Self {
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

impl PitchHalver {
    fn new() -> Self {
        let mut real_planner = RealFftPlanner::new();
        PitchHalver {
            input_buffer: SegQueue::new(),
            output_buffer: SegQueue::new(),
            forward_fft: real_planner.plan_fft_forward(BUFFER_SIZE),
            inverse_fft: real_planner.plan_fft_inverse(BUFFER_SIZE),
        }
    }

    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum = self.forward_fft.make_output_vec();
        self.forward_fft.process(buffer, &mut spectrum).unwrap();
        // TODO: Some phase manipulation
        self.inverse_fft.process(&mut spectrum, buffer).unwrap();
        let mut temp = [0.0; BUFFER_SIZE / 2].to_vec();
        temp.clone_from_slice(&buffer[0..BUFFER_SIZE / 2]);
        for (index, sample) in buffer.iter_mut().enumerate() {
            *sample = temp[index / 2];
        }
    }
}

impl HighPassFilter {
    fn new() -> Self {
        let mut real_planner = RealFftPlanner::new();
        Self {
            input_buffer: SegQueue::new(),
            output_buffer: SegQueue::new(),
            forward_fft: real_planner.plan_fft_forward(BUFFER_SIZE),
            inverse_fft: real_planner.plan_fft_inverse(BUFFER_SIZE),
        }
    }

    fn process(&self, buffer: &mut [f32]) {
        let mut spectrum = self.forward_fft.make_output_vec();
        self.forward_fft.process(buffer, &mut spectrum).unwrap();
        spectrum[0..15]
            .iter_mut()
            .for_each(|sample| *sample = Complex::new(0.0, 0.0));
        self.inverse_fft.process(&mut spectrum, buffer).unwrap();
    }
}

impl StreamProcessor for PitchHalver {
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
            self.process(&mut buffer);
            for sample in buffer {
                self.output_buffer.push(sample / BUFFER_SIZE as f32);
            }
        }
    }
}

impl StreamProcessor for HighPassFilter {
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
            self.process(&mut buffer);
            for sample in buffer {
                self.output_buffer.push(sample / BUFFER_SIZE as f32);
            }
        }
    }
}

fn simple_pitch_halver() {
    let composed_processor =
        ComposedProcessor::new(DisplayProcessor::new(true), PitchHalver::new());
    let composed_processor = ComposedProcessor::new(composed_processor, PitchHalver::new());
    let composed_processor =
        ComposedProcessor::new(composed_processor, DisplayProcessor::new(false));
    let _streams = hardware::setup_passthrough_processor(composed_processor);
    std::thread::park();
}

fn high_pass_filter() {
    let composed_processor =
        ComposedProcessor::new(DisplayProcessor::new(true), HighPassFilter::new());
    let composed_processor =
        ComposedProcessor::new(composed_processor, DisplayProcessor::new(false));
    let _streams = hardware::setup_passthrough_processor(composed_processor);
    std::thread::park();
}

fn main() {
    let opts: Opts = Opts::parse();
    ctrlc::set_handler(move || {
        print!("{}", ansi_escapes::CursorTo::AbsoluteX(0));
        print!("{}", ansi_escapes::ClearScreen);
        print!("{}", termion::cursor::Show);
        std::process::exit(130);
    })
    .expect("Error setting Ctrl-C handler");

    print!("{}", ansi_escapes::ClearScreen);
    print!("{}", termion::cursor::Hide);

    match opts.subcmd {
        SubCommand::Passthrough => passthrough(),
        SubCommand::SimplePitchHalver => simple_pitch_halver(),
        SubCommand::HighPassFilter => high_pass_filter(),
    }
}
