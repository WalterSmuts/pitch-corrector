use clap::Parser;
use cpal::Stream;
use pitch_corrector::display::UserInterface;
use pitch_corrector::hardware;
use pitch_corrector::pitch_correction;
use pitch_corrector::signal_processing::pipeline;
use pitch_corrector::signal_processing::FrequencyDomainPitchShifter;
use pitch_corrector::signal_processing::HighPassFilter;
use pitch_corrector::signal_processing::LowPassFilter;
use pitch_corrector::signal_processing::NaivePitchShifter;
use pitch_corrector::signal_processing::OverlapAndAddProcessor;
use pitch_corrector::signal_processing::PhaseVocoderPitchShifter;
use pitch_corrector::signal_processing::Segmenter;
use pitch_corrector::signal_processing::TimeToFrequencyDomainBlockProcessorConverter;

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
        SubCommand::PitchCorrector => {
            use pitch_corrector::music::{Note, Scale};

            let corrector = pitch_correction::PitchCorrector::new();
            let controls = corrector.controls();
            let status = user_inferface.status_handle();
            let _streams = hardware::setup_passthrough_processor(pipeline!(
                user_inferface.create_display_processor(),
                corrector,
                user_inferface.create_display_processor(),
            ));

            let scale_presets: Vec<(&str, Scale)> = vec![
                ("Off", Scale::empty()),
                ("Chromatic", Scale::chromatic()),
                ("C Major", Scale::major(Note::C)),
                ("C Minor", Scale::minor(Note::C)),
                ("C Pentatonic", Scale::pentatonic(Note::C)),
                ("G Major", Scale::major(Note::G)),
                ("A Minor", Scale::minor(Note::A)),
            ];
            let mut scale_idx: usize = 4; // Start on C Pentatonic

            let update_status = {
                let status = status.clone();
                move |scale_name: &str, semitones: f32| {
                    *status.lock().unwrap() = format!(
                        " Scale: {} | Shift: {:.0} semitones | [S]cale [Up/Down]shift [0]reset [L]ogger",
                        scale_name, semitones
                    );
                }
            };
            update_status(scale_presets[scale_idx].0, 0.0);

            log_panics::init();
            user_inferface.run_with_key_handler(move |key| {
                use crossterm::event::KeyCode;

                match key {
                    KeyCode::Up => {
                        let s = controls.get_shift() + 1.0;
                        controls.set_shift(s);
                        update_status(scale_presets[scale_idx].0, s);
                    }
                    KeyCode::Down => {
                        let s = controls.get_shift() - 1.0;
                        controls.set_shift(s);
                        update_status(scale_presets[scale_idx].0, s);
                    }
                    KeyCode::Char('s') => {
                        scale_idx = (scale_idx + 1) % scale_presets.len();
                        controls.set_scale(scale_presets[scale_idx].1);
                        update_status(scale_presets[scale_idx].0, controls.get_shift());
                    }
                    KeyCode::Char('0') => {
                        controls.set_shift(0.0);
                        update_status(scale_presets[scale_idx].0, 0.0);
                    }
                    _ => {}
                }
            });
            return;
        }
    };
    log_panics::init();
    user_inferface.run();
}
