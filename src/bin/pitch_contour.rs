//! Offline pitch contour extraction using pyin-rs (probabilistic YIN + HMM).
//!
//! Reads a WAV file, runs the pYIN algorithm, and writes CSV
//! (time_seconds, frequency_hz) to stdout. Unvoiced frames are omitted.

use hound::WavReader;
use pyin::Framing;
use pyin::PYINExecutor;
use pyin::PadMode;
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: pitch_contour <input.wav>");
        process::exit(1);
    }
    let path = &args[1];

    let reader = WavReader::open(path).unwrap_or_else(|e| {
        eprintln!("Cannot open {}: {}", path, e);
        process::exit(1);
    });
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    eprintln!(
        "WAV: {}Hz, {} ch, {} bit",
        sample_rate, channels, spec.bits_per_sample
    );

    // Read mono samples as f64 (pyin-rs uses f64)
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .into_samples::<i32>()
                .step_by(channels)
                .map(|s| s.unwrap() as f64 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .step_by(channels)
            .map(|s| s.unwrap() as f64)
            .collect(),
    };

    eprintln!(
        "{} samples ({:.2}s)",
        samples.len(),
        samples.len() as f64 / sample_rate as f64
    );

    // Run pYIN
    eprintln!("Running pYIN...");
    let fmin = 80.0_f64;
    let fmax = 300.0_f64;
    let frame_length = 2048_usize;

    let mut pyin_exec = PYINExecutor::new(
        fmin,
        fmax,
        sample_rate,
        frame_length,
        None, // win_length (default: frame_length / 2)
        None, // hop_length (default: frame_length / 4)
        None, // resolution (default: 0.1)
    );

    let framing = Framing::Center(PadMode::Constant(0.0));
    let fill_unvoiced = f64::NAN;

    let (timestamp, f0, voiced_flag, _voiced_prob) =
        pyin_exec.pyin(&samples, fill_unvoiced, framing);

    // Output voiced frames
    println!("time_s,frequency_hz");
    let mut voiced_count = 0;
    for i in 0..f0.len() {
        if voiced_flag[i] && !f0[i].is_nan() && f0[i] > 0.0 {
            println!("{:.4},{:.2}", timestamp[i], f0[i]);
            voiced_count += 1;
        }
    }
    eprintln!("{} voiced frames output", voiced_count);
}
