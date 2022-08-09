use clap::Parser;
use cpal::traits::StreamTrait;
use cpal::Sample;
use cpal::Stream;
use display::UserInterface;
use signal_processing::compose;
use signal_processing::ComposedProcessor;
use signal_processing::FrequencyDomainPitchShifter;
use signal_processing::HighPassFilter;
use signal_processing::LowPassFilter;
use signal_processing::NaivePitchShifter;
use signal_processing::Segmenter;
use signal_processing::StreamProcessor;
use std::f32::consts::TAU;
use std::sync::Arc;
use std::sync::Barrier;

mod complex_interpolation;
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

fn passthrough(user_interface: &mut UserInterface) -> (Stream, Stream) {
    let display_processor = user_interface.create_display_processor();
    hardware::setup_passthrough_processor(display_processor)
}

fn naive_pitch_shifter(user_interface: &mut UserInterface) -> (Stream, Stream) {
    hardware::setup_passthrough_processor(compose(
        compose(
            user_interface.create_display_processor(),
            Segmenter::new(NaivePitchShifter::new(1.2)),
        ),
        user_interface.create_display_processor(),
    ))
}

fn high_pass_filter(user_interface: &mut UserInterface) -> (Stream, Stream) {
    let display_processor = user_interface.create_display_processor();
    let composed_processor =
        ComposedProcessor::new(display_processor, Segmenter::new(HighPassFilter::new()));
    let display_processor = user_interface.create_display_processor();
    let composed_processor = ComposedProcessor::new(composed_processor, display_processor);
    hardware::setup_passthrough_processor(composed_processor)
}

fn low_pass_filter(user_interface: &mut UserInterface) -> (Stream, Stream) {
    let display_processor = user_interface.create_display_processor();
    let composed_processor =
        ComposedProcessor::new(display_processor, Segmenter::new(LowPassFilter::new()));
    let display_processor = user_interface.create_display_processor();
    let composed_processor = ComposedProcessor::new(composed_processor, display_processor);
    hardware::setup_passthrough_processor(composed_processor)
}

fn frequency_domain_pitch_shifter(user_inferface: &mut UserInterface) -> (Stream, Stream) {
    let display_processor = user_inferface.create_display_processor();
    let composed_processor = ComposedProcessor::new(
        display_processor,
        Segmenter::new(FrequencyDomainPitchShifter::new()),
    );
    let display_processor = user_inferface.create_display_processor();
    let composed_processor = ComposedProcessor::new(composed_processor, display_processor);
    hardware::setup_passthrough_processor(composed_processor)
}

const SAMPLE_RATE: usize = 44100;

fn play(user_inferface: &mut UserInterface) -> (Stream, Stream) {
    let barrier = Arc::new(Barrier::new(2));
    let barrier_clone = barrier.clone();
    let once = std::sync::Once::new();
    let display_processor = user_inferface.create_display_processor();
    let pitch_halver = ComposedProcessor::new(
        Segmenter::new(FrequencyDomainPitchShifter::new()),
        display_processor,
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
    barrier.wait(); // TODO: FIX this return
    todo!()
}

fn main() {
    tui_logger::init_logger(log::LevelFilter::Trace).unwrap();
    let opts: Opts = Opts::parse();
    let mut user_inferface = UserInterface::new();

    // Don't drop streams otherwize we drop the threads doing the data processing
    let _streams = match opts.subcmd {
        SubCommand::Passthrough => passthrough(&mut user_inferface),
        SubCommand::NaivePitchShifter => naive_pitch_shifter(&mut user_inferface),
        SubCommand::HighPassFilter => high_pass_filter(&mut user_inferface),
        SubCommand::LowPassFilter => low_pass_filter(&mut user_inferface),
        SubCommand::FrequencyDomainPitchShifter => {
            frequency_domain_pitch_shifter(&mut user_inferface)
        }
        SubCommand::Play => play(&mut user_inferface),
    };
    user_inferface.run();
}
