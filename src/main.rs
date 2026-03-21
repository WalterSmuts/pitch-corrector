#[cfg(test)]
#[global_allocator]
static A: assert_no_alloc::AllocDisabler = assert_no_alloc::AllocDisabler;
use clap::Parser;
use cpal::Stream;
use display::UserInterface;
use signal_processing::pipeline;
use signal_processing::FrequencyDomainPitchShifter;
use signal_processing::HighPassFilter;
use signal_processing::LowPassFilter;
use signal_processing::NaivePitchShifter;
use signal_processing::OverlapAndAddProcessor;
use signal_processing::PhaseVocoderPitchShifter;
use signal_processing::Segmenter;
use signal_processing::TimeToFrequencyDomainBlockProcessorConverter;

mod complex_interpolation;
mod display;
mod hardware;
mod interpolation;
mod pitch_correction;
mod signal_processing;

const FILTER_CUTOFF_FREQUENCY: usize = 440;

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
    FrequencyDomainPitchShifter {
        /// Pitch scaling ratio (e.g. 0.5 = down one octave, 2.0 = up one octave)
        #[clap(default_value = "0.5")]
        ratio: f32,
    },
    /// Pitch shift using phase vocoder (better quality)
    PhaseVocoder {
        /// Pitch scaling ratio (e.g. 0.5 = down one octave, 2.0 = up one octave)
        #[clap(default_value = "0.5")]
        ratio: f32,
    },
    /// Auto-tune: detect pitch and correct to nearest scale note
    PitchCorrector,
}

fn passthrough(user_interface: &mut UserInterface) -> (Stream, Stream) {
    let display_processor = user_interface.create_display_processor();
    hardware::setup_passthrough_processor(display_processor)
}

fn naive_pitch_shifter(user_interface: &mut UserInterface) -> (Stream, Stream) {
    hardware::setup_passthrough_processor(pipeline!(
        user_interface.create_display_processor(),
        Segmenter::new(NaivePitchShifter::new(1.2)),
        user_interface.create_display_processor(),
    ))
}

fn high_pass_filter(user_interface: &mut UserInterface) -> (Stream, Stream) {
    hardware::setup_passthrough_processor(pipeline!(
        user_interface.create_display_processor(),
        Segmenter::new(OverlapAndAddProcessor::new(
            TimeToFrequencyDomainBlockProcessorConverter::new(HighPassFilter::new(
                FILTER_CUTOFF_FREQUENCY
            ))
        )),
        user_interface.create_display_processor(),
    ))
}

fn low_pass_filter(user_interface: &mut UserInterface) -> (Stream, Stream) {
    hardware::setup_passthrough_processor(pipeline!(
        user_interface.create_display_processor(),
        Segmenter::new(OverlapAndAddProcessor::new(
            TimeToFrequencyDomainBlockProcessorConverter::new(LowPassFilter::new(
                FILTER_CUTOFF_FREQUENCY
            ))
        )),
        user_interface.create_display_processor(),
    ))
}

fn frequency_domain_pitch_shifter(
    user_interface: &mut UserInterface,
    ratio: f32,
) -> (Stream, Stream) {
    hardware::setup_passthrough_processor(pipeline!(
        user_interface.create_display_processor(),
        Segmenter::new(OverlapAndAddProcessor::new(
            TimeToFrequencyDomainBlockProcessorConverter::new(FrequencyDomainPitchShifter::new(
                ratio
            ))
        )),
        user_interface.create_display_processor(),
    ))
}

fn phase_vocoder(user_interface: &mut UserInterface, ratio: f32) -> (Stream, Stream) {
    hardware::setup_passthrough_processor(pipeline!(
        user_interface.create_display_processor(),
        PhaseVocoderPitchShifter::new(ratio),
        user_interface.create_display_processor(),
    ))
}

fn pitch_corrector(user_interface: &mut UserInterface) -> (Stream, Stream) {
    hardware::setup_passthrough_processor(pipeline!(
        user_interface.create_display_processor(),
        pitch_correction::new_pitch_corrector(),
        user_interface.create_display_processor(),
    ))
}

fn main() {
    tui_logger::init_logger(log::LevelFilter::Trace).unwrap();
    tui_logger::set_default_level(log::LevelFilter::Trace);
    let opts: Opts = Opts::parse();
    let mut user_inferface = UserInterface::new();

    // Don't drop streams otherwize we drop the threads doing the data processing
    let _streams = match opts.subcmd {
        SubCommand::Passthrough => passthrough(&mut user_inferface),
        SubCommand::NaivePitchShifter => naive_pitch_shifter(&mut user_inferface),
        SubCommand::HighPassFilter => high_pass_filter(&mut user_inferface),
        SubCommand::LowPassFilter => low_pass_filter(&mut user_inferface),
        SubCommand::FrequencyDomainPitchShifter { ratio } => {
            frequency_domain_pitch_shifter(&mut user_inferface, ratio)
        }
        SubCommand::PhaseVocoder { ratio } => phase_vocoder(&mut user_inferface, ratio),
        SubCommand::PitchCorrector => pitch_corrector(&mut user_inferface),
    };
    log_panics::init();
    user_inferface.run();
}
