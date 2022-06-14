use clap::Parser;
use crossbeam_queue::SegQueue;
use display::SignalDrawer;
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
}
fn passthrough() {
    let _streams = hardware::setup_passthrough_processor(DisplayProcessor::new());
    std::thread::park();
}

struct PitchHalver {
    input_buffer: SegQueue<f32>,
    output_buffer: SegQueue<f32>,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
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
    fn new() -> Self {
        Self {
            buffer: SegQueue::new(),
            display_buffer: Mutex::new(Box::new([0.0; BUFFER_SIZE])),
            buffer_index: AtomicUsize::new(0),
            signal_drawer: SignalDrawer::new(),
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

fn simple_pitch_halver() {
    let _streams = hardware::setup_passthrough_processor(PitchHalver::new());
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
    }
}
