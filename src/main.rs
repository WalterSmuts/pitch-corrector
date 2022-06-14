use clap::Parser;
use signal_processing::ComposedProcessor;
use signal_processing::DisplayProcessor;
use signal_processing::FrequencyDomainPitchShifter;
use signal_processing::HighPassFilter;
use signal_processing::LowPassFilter;
use signal_processing::NaivePitchHalver;
use signal_processing::Segmenter;

mod display;
mod hardware;
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
    NaivePitchHalver,
    /// Passthrough microphone to speakers but filter out low frequencies
    HighPassFilter,
    /// Passthrough microphone to speakers but filter out high frequencies
    LowPassFilter,
    /// Passthrough microphone to speakers but shift pitch in the frequency domain
    FrequencyDomainPitchShifter,
}

fn passthrough() {
    let _streams = hardware::setup_passthrough_processor(DisplayProcessor::new(true));
    std::thread::park();
}

fn naive_pitch_halver() {
    let composed_processor = ComposedProcessor::new(
        DisplayProcessor::new(true),
        Segmenter::new(NaivePitchHalver),
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
        SubCommand::NaivePitchHalver => naive_pitch_halver(),
        SubCommand::HighPassFilter => high_pass_filter(),
        SubCommand::LowPassFilter => low_pass_filter(),
        SubCommand::FrequencyDomainPitchShifter => frequency_domain_pitch_shifter(),
    }
}
