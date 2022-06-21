use clap::Parser;
use cpal::traits::StreamTrait;
use cpal::Sample;
use signal_processing::ComposedProcessor;
use signal_processing::DisplayProcessor;
use signal_processing::FrequencyDomainPitchShifter;
use signal_processing::HighPassFilter;
use signal_processing::LowPassFilter;
use signal_processing::NaivePitchShifter;
use signal_processing::Segmenter;
use signal_processing::StreamProcessor;
use std::f32::consts::TAU;
use std::sync::Arc;
use std::sync::Barrier;

mod display;
mod hardware;
mod interpolation;
mod signal_processing;

#[derive(Parser)]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    Passthrough,
    /// Passthrough microphone to speakers but halve the pitch
    NaivePitchShifter,
    /// Passthrough microphone to speakers but filter out low frequencies
    HighPassFilter,
    /// Passthrough microphone to speakers but filter out high frequencies
    LowPassFilter,
    /// Passthrough microphone to speakers but shift pitch in the frequency domain
    FrequencyDomainPitchShifter,
    /// Play sine wave
    Play,
}

fn passthrough() {
    let _streams = hardware::setup_passthrough_processor(DisplayProcessor::new(true));
    std::thread::park();
}

fn naive_pitch_shifter() {
    let composed_processor = ComposedProcessor::new(
        DisplayProcessor::new(true),
        Segmenter::new(NaivePitchShifter::new(1.2)),
    );
    let composed_processor =
        ComposedProcessor::new(composed_processor, DisplayProcessor::new(false));
    let _streams = hardware::setup_passthrough_processor(composed_processor);
    std::thread::park();
}

fn high_pass_filter() {
    let composed_processor = ComposedProcessor::new(
        DisplayProcessor::new(true),
        Segmenter::new(HighPassFilter::new()),
    );
    let composed_processor =
        ComposedProcessor::new(composed_processor, DisplayProcessor::new(false));
    let _streams = hardware::setup_passthrough_processor(composed_processor);
    std::thread::park();
}

fn low_pass_filter() {
    let composed_processor = ComposedProcessor::new(
        DisplayProcessor::new(true),
        Segmenter::new(LowPassFilter::new()),
    );
    let composed_processor =
        ComposedProcessor::new(composed_processor, DisplayProcessor::new(false));
    let _streams = hardware::setup_passthrough_processor(composed_processor);
    std::thread::park();
}

fn frequency_domain_pitch_shifter() {
    let composed_processor = ComposedProcessor::new(
        DisplayProcessor::new(true),
        Segmenter::new(FrequencyDomainPitchShifter::new()),
    );
    let composed_processor =
        ComposedProcessor::new(composed_processor, DisplayProcessor::new(false));
    let _streams = hardware::setup_passthrough_processor(composed_processor);
    std::thread::park();
}

const SAMPLE_RATE: usize = 44100;

fn play() {
    let barrier = Arc::new(Barrier::new(2));
    let barrier_clone = barrier.clone();
    let once = std::sync::Once::new();

    let pitch_halver = ComposedProcessor::new(
        Segmenter::new(FrequencyDomainPitchShifter::new()),
        DisplayProcessor::new(true),
    );

    for t in (0..SAMPLE_RATE * 5).map(|x| x as f32 / SAMPLE_RATE as f32) {
        let sample = (TAU * t * 440.0).sin();
        pitch_halver.push_sample(sample);
    }

    let stream = hardware::get_output_stream(move |data: &mut [f32], _| {
        for datum in data.iter_mut() {
            if let Some(sample) = pitch_halver.pop_sample() {
                *datum = Sample::from(&sample);
            } else {
                once.call_once(|| {
                    barrier_clone.wait();
                });
            }
        }
    });
    stream.play().unwrap();
    barrier.wait();
}

fn main() {
    let opts: Opts = Opts::parse();
    ctrlc::set_handler(move || {
        reset_screen();
        std::process::exit(130);
    })
    .expect("Error setting Ctrl-C handler");

    print!("{}", ansi_escapes::ClearScreen);
    print!("{}", termion::cursor::Hide);

    match opts.subcmd {
        SubCommand::Passthrough => passthrough(),
        SubCommand::NaivePitchShifter => naive_pitch_shifter(),
        SubCommand::HighPassFilter => high_pass_filter(),
        SubCommand::LowPassFilter => low_pass_filter(),
        SubCommand::FrequencyDomainPitchShifter => frequency_domain_pitch_shifter(),
        SubCommand::Play => play(),
    }
    reset_screen();
}

fn reset_screen() {
    print!("{}", ansi_escapes::CursorTo::AbsoluteX(0));
    print!("{}", ansi_escapes::ClearScreen);
    print!("{}", termion::cursor::Show);
}
