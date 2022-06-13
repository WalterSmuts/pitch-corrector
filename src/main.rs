use clap::Parser;
use cpal::{InputCallbackInfo, OutputCallbackInfo};
use realfft::RealToComplex;
use realfft::{ComplexToReal, RealFftPlanner};
use splines::{Interpolation, Key, Spline};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::BufferSize;
use cpal::Sample;
use cpal::{SampleRate, StreamConfig};
use crossbeam_queue::SegQueue;
use std::f32::consts::PI;
use std::sync::{Arc, Barrier};
use textplots::Shape;
use textplots::{Chart, Plot};

const TAU: f32 = 2.0 * PI;
const BUFFER_SIZE: usize = 1024;
const SAMPLE_RATE: u32 = 44100;

#[derive(Parser)]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
    /// Sets the name of the input/output wav file
    #[clap(short, long, default_value = "example.wav")]
    filename: String,
}

#[derive(Parser)]
enum SubCommand {
    /// Generate a wav file
    Write,
    /// Analize a wav file
    Read,
    /// Play wav file and showing live levels
    Play,
    /// Record from microphone while showing live levels
    Record,
    /// Record from microphone and play directly to speakers
    Passthrough,
    /// Passthrough microphone to speakers but halve the pitch
    SimplePitchHalver,
}

fn read(filename: &String) {
    let mut reader = hound::WavReader::open(filename).unwrap();
    let sqr_sum = reader.samples::<i16>().fold(0.0, |sqr_sum, s| {
        let sample = s.unwrap() as f64;
        sqr_sum + sample * sample
    });
    println!("RMS is {}", (sqr_sum / reader.len() as f64).sqrt());
}

fn write(filename: &String) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();
    for t in (0..SAMPLE_RATE * 5).map(|x| x as f32 / SAMPLE_RATE as f32) {
        let sample = (TAU * t * 440.0).sin();
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

fn play(filename: &String) {
    let mut reader = hound::WavReader::open(filename).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let barrier_clone = barrier.clone();
    let once = std::sync::Once::new();

    let stream = get_output_stream(move |data: &mut [f32], _| {
        for sample in data.iter_mut() {
            *sample = Sample::from(
                &reader
                    .samples::<i16>()
                    .map(|sample| sample.unwrap())
                    .next()
                    .unwrap_or_else(|| {
                        once.call_once(|| {
                            barrier_clone.wait();
                        });
                        0
                    }),
            );
        }
        draw_data(data);
    });
    stream.play().unwrap();
    barrier.wait();
}

fn record() {
    let stream = get_input_stream(move |data: &[f32], _| {
        draw_data(data);
    });
    stream.play().unwrap();
    std::thread::park();
}

fn passthrough() {
    setup_passthrough_processor(DisplayProcessor)
}

struct PitchHalver {
    input_buffer: SegQueue<f32>,
    output_buffer: SegQueue<f32>,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

struct DisplayProcessor;

trait StreamProcessor {
    fn push_sample(&self, sample: f32);
    fn pop_sample(&self) -> Option<f32>;
}

impl StreamProcessor for DisplayProcessor {
    fn push_sample(&self, sample: f32) {
        todo!()
    }

    fn pop_sample(&self) -> Option<f32> {
        todo!()
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
    setup_passthrough_processor(PitchHalver::new());
}

fn setup_passthrough_processor<T: 'static>(processor: T)
where
    T: StreamProcessor + Send + Sync,
{
    let input_passthrough_processor = Arc::new(processor);
    let output_passthrough_processor = input_passthrough_processor.clone();

    let input_stream = get_input_stream(move |data: &[f32], _| {
        draw_data(data);
        for datum in data {
            input_passthrough_processor.push_sample(*datum);
        }
    });
    input_stream.play().unwrap();

    let output_stream = get_output_stream(move |data: &mut [f32], _| {
        for sample in data.iter_mut() {
            *sample = Sample::from(&output_passthrough_processor.pop_sample().unwrap_or(0.0));
        }
    });
    output_stream.play().unwrap();
    std::thread::park();
}

fn get_input_stream<T, D>(handler: D) -> cpal::Stream
where
    T: Sample,
    D: FnMut(&[T], &InputCallbackInfo) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");

    device
        .build_input_stream(
            &StreamConfig {
                channels: 1,
                sample_rate: SampleRate(SAMPLE_RATE),
                buffer_size: BufferSize::Fixed((BUFFER_SIZE * 4) as u32),
            },
            handler,
            |_| panic!("Error from ALSA on input"),
        )
        .unwrap()
}

fn get_output_stream<T, D>(handler: D) -> cpal::Stream
where
    T: Sample,
    D: FnMut(&mut [T], &OutputCallbackInfo) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No input device available");

    device
        .build_output_stream(
            &StreamConfig {
                channels: 1,
                sample_rate: SampleRate(SAMPLE_RATE),
                buffer_size: BufferSize::Fixed((BUFFER_SIZE * 4) as u32),
            },
            handler,
            |_| panic!("Error from ALSA on output"),
        )
        .unwrap()
}

fn draw_waveform(data: &[f32]) {
    println!("Waveform:");
    let (width, height) = get_textplots_window_size();
    let vec = data
        .iter()
        .enumerate()
        .map(|(index, sample)| Key::new(index as f32, *sample as f32, Interpolation::Linear))
        .collect();

    let spline = Spline::from_vec(vec);
    Chart::new_with_y_range(width, height, 0.0, BUFFER_SIZE as f32, -1.0, 1.0)
        .lineplot(&Shape::Continuous(Box::new(|x| {
            spline.sample(x).unwrap_or(0.0)
        })))
        .display();
}

fn draw_psd(data: &[f32]) {
    println!("Power Spectral Density:");
    let (width, height) = get_textplots_window_size();

    let mut real_planner = RealFftPlanner::<f32>::new();
    let r2c = real_planner.plan_fft_forward(BUFFER_SIZE);
    let mut ff_data = [0.0; BUFFER_SIZE].to_vec();
    ff_data.copy_from_slice(&data[0..BUFFER_SIZE]);
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut ff_data, &mut spectrum).unwrap();

    let vec: Vec<_> = spectrum
        .into_iter()
        .map(|complex| complex.norm_sqr())
        .enumerate()
        .map(|(index, val)| ((index as f32).log2() * 115.4, val))
        .map(|(index, sample)| Key::new(index as f32, sample as f32, Interpolation::Cosine))
        .collect();

    let spline = Spline::from_vec(vec);

    Chart::new_with_y_range(width, height, 0.0, BUFFER_SIZE as f32, 0.0, 50.0)
        .lineplot(&Shape::Continuous(Box::new(|x| {
            spline.sample(x).unwrap_or(0.0)
        })))
        .display();
}

fn get_textplots_window_size() -> (u32, u32) {
    let (mut width, height) = termion::terminal_size().unwrap();
    width = width * 2 - 11;
    let height = height as f32 * 1.6;
    (width as u32, height as u32)
}

fn draw_data(data: &[f32]) {
    print!("{}", ansi_escapes::CursorTo::TopLeft);
    draw_psd(data);
    draw_waveform(data);
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
        SubCommand::Write => write(&opts.filename),
        SubCommand::Read => read(&opts.filename),
        SubCommand::Play => play(&opts.filename),
        SubCommand::Record => record(),
        SubCommand::Passthrough => passthrough(),
        SubCommand::SimplePitchHalver => simple_pitch_halver(),
    }
}
