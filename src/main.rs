use std::env;
use std::f32::consts::TAU;
use std::i16;
use std::process;
use hound;

const USAGE: &str = "Usage: pitch-corrector <action> <filename> # Where <action> is either write or read";

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
    for t in (0 .. 44100 * 5).map(|x| x as f32 / 44100.0) {
        let sample = (TAU * t * 440.0).sin();
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        println!("{}", USAGE);
        process::exit(1);
    }

    let action = &args[1];

    match action.as_str() {
        "write" => write(&args[2]),
        "read"  => read(&args[2]),
        _ => println!("{}", USAGE),
    }
}
