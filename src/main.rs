use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample;
use cpal::{SampleRate, StreamConfig};
use hound;
use std::f32::consts::PI;
use std::sync::{Arc, Barrier};

const TAU: f32 = 2.0 * PI;

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
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();
    for t in (0..44100 * 5).map(|x| x as f32 / 44100.0) {
        let sample = (TAU * t * 440.0).sin();
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

fn play(filename: &String) {
    let mut reader = hound::WavReader::open(filename).unwrap();
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device available");

    let barrier = Arc::new(Barrier::new(2));
    let barrier_clone = barrier.clone();
    let once = std::sync::Once::new();

    let stream = device
        .build_output_stream(
            &StreamConfig {
                channels: 1,
                sample_rate: SampleRate(44100),
                buffer_size: cpal::BufferSize::Default,
            },
            move |data: &mut [f32], _| {
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
            },
            |_| panic!("Error from ALSA"),
        )
        .unwrap();
    stream.play().unwrap();
    barrier.wait();
}

fn main() {
    let opts: Opts = Opts::parse();

    match opts.subcmd {
        SubCommand::Write => write(&opts.filename),
        SubCommand::Read => read(&opts.filename),
        SubCommand::Play => play(&opts.filename),
    }
}
