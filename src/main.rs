use clap::Parser;
use cpal::{InputCallbackInfo, OutputCallbackInfo};
use realfft::RealFftPlanner;

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
    PassThrough,
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
        draw_data(&data);
    });
    stream.play().unwrap();
    barrier.wait();
}

fn record() {
    let stream = get_input_stream(move |data: &[f32], _| {
        draw_data(&data);
    });
    stream.play().unwrap();
    std::thread::park();
}

fn pass_through() {
    let input_buffer: Arc<SegQueue<f32>> = Arc::new(SegQueue::new());
    let output_buffer = input_buffer.clone();

    let input_stream = get_input_stream(move |data: &[f32], _| {
        draw_data(&data);
        for datum in data {
            input_buffer.push(*datum);
        }
    });
    input_stream.play().unwrap();

    let output_stream = get_output_stream(move |data: &mut [f32], _| {
        for sample in data.iter_mut() {
            *sample = Sample::from(&output_buffer.pop().unwrap_or_else(|| 0.0));
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
    let input_stream = device
        .build_input_stream(
            &StreamConfig {
                channels: 1,
                sample_rate: SampleRate(SAMPLE_RATE),
                buffer_size: BufferSize::Fixed((BUFFER_SIZE * 4) as u32),
            },
            handler,
            |_| panic!("Error from ALSA on input"),
        )
        .unwrap();
    input_stream
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
    let output_stream = device
        .build_output_stream(
            &StreamConfig {
                channels: 1,
                sample_rate: SampleRate(SAMPLE_RATE),
                buffer_size: BufferSize::Fixed((BUFFER_SIZE * 4) as u32),
            },
            handler,
            |_| panic!("Error from ALSA on output"),
        )
        .unwrap();
    output_stream
}

fn draw_waveform(data: &[f32]) {
    let mut index = -(BUFFER_SIZE as i32) / 2;
    let mut points = Vec::with_capacity(BUFFER_SIZE);
    for point in data {
        points.push((index as f32, *point));
        if points.len() == BUFFER_SIZE {
            break;
        }
        index += 1;
    }
    Chart::new_with_y_range(
        200,
        100,
        -(BUFFER_SIZE as f32) / 2.0,
        BUFFER_SIZE as f32 / 2.0,
        -1.0,
        1.0,
    )
    .lineplot(&Shape::Points(&points))
    .display();
}

fn draw_psd(data: &[f32]) {
    let mut real_planner = RealFftPlanner::<f32>::new();
    let r2c = real_planner.plan_fft_forward(BUFFER_SIZE);
    let mut ff_data = [0.0; BUFFER_SIZE].to_vec();
    ff_data.copy_from_slice(&data[0..BUFFER_SIZE]);
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut ff_data, &mut spectrum).unwrap();
    let spectrum: Vec<f32> = spectrum
        .into_iter()
        .map(|complex| complex.norm_sqr())
        .collect();
    let spectrum: Vec<(f32, f32)> = spectrum
        .into_iter()
        .enumerate()
        .map(|(index, val)| ((index as f32).log2() * 102.4, val))
        .collect();
    Chart::new_with_y_range(200, 100, 0.0, BUFFER_SIZE as f32, 0.0, 30.0)
        .lineplot(&Shape::Points(&spectrum))
        .display();
}

fn draw_data(data: &[f32]) {
    print!("{}", ansi_escapes::CursorTo::TopLeft);
    draw_psd(data);
    draw_waveform(data);
}

fn main() {
    print!("{}[2J", 27 as char);
    let opts: Opts = Opts::parse();

    match opts.subcmd {
        SubCommand::Write => write(&opts.filename),
        SubCommand::Read => read(&opts.filename),
        SubCommand::Play => play(&opts.filename),
        SubCommand::Record => record(),
        SubCommand::PassThrough => pass_through(),
    }
}
