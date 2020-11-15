use std::f32::consts::TAU;
use std::i16;
use hound;
use clap::Clap;

#[derive(Clap)]
#[clap()]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
    /// Sets the name of the input/output wav file
    #[clap(short, long, default_value = "example.wav")]
    filename: String,
}

#[derive(Clap)]
enum SubCommand {
    /// Generate a wav file
    #[clap()]
    Write,
    /// Analize a wav file
    #[clap()]
    Read,
}

fn read(filename: &String) {
    let mut reader = hound::WavReader::open(filename).unwrap();
    let sqr_sum = reader.samples::<i16>().fold(0.0, |sqr_sum, s| {
        let sample = s.unwrap() as f64;
        sqr_sum + sample * sample
    });
    println!("RMS is {}", (sqr_sum / reader.len() as f64).sqrt());
}

fn write(filename: &String) { let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();
    for t in (0 .. 44100 * 5).map(|x| x as f32 / 44100.0) {
        let sample = (TAU * t * 440.0).sin();
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

fn main() {
    let opts: Opts = Opts::parse();

    match opts.subcmd {
        SubCommand::Write => write(&opts.filename),
        SubCommand::Read  =>  read(&opts.filename),
    }
}
