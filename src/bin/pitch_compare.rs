//! Compare our YIN pitch detection against pyin-rs (pYIN with HMM).
//! Outputs two CSV files: one for each algorithm.

use hound::WavReader;
use pitch_corrector::signal_processing::BUFFER_SIZE;
use pitch_corrector::signal_processing::YinPitchDetector;
use pyin::Framing;
use pyin::PYINExecutor;
use pyin::PadMode;
use std::env;
use std::fs::File;
use std::io::Write;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: pitch_compare <input.wav>");
        process::exit(1);
    }
    let path = &args[1];

    // Read WAV using hound
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

    let samples_i: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .step_by(channels)
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .step_by(channels)
            .map(|s| s.unwrap())
            .collect(),
    };

    let duration = samples_i.len() as f64 / sample_rate as f64;
    eprintln!("{} samples ({:.2}s)", samples_i.len(), duration);

    // --- Our YIN ---
    eprintln!("Running our YIN...");
    let mut detector = YinPitchDetector::with_sample_rate(sample_rate as f32);
    let hop = BUFFER_SIZE / 2;
    let mut ours: Vec<(f64, f32)> = Vec::new();
    let mut pos = 0;
    while pos + BUFFER_SIZE <= samples_i.len() {
        let frame = &samples_i[pos..pos + BUFFER_SIZE];
        if let Some(freq) = detector.detect(frame) {
            let time = (pos + BUFFER_SIZE / 2) as f64 / sample_rate as f64;
            ours.push((time, freq));
        }
        pos += hop;
    }
    eprintln!("  Our YIN: {} voiced frames", ours.len());

    // --- pyin-rs ---
    eprintln!("Running pyin-rs...");
    let fmin = 80.0_f64;
    let fmax = 300.0_f64;
    let frame_length = 2048_usize;

    let mut pyin_exec = PYINExecutor::new(
        fmin,
        fmax,
        sample_rate,
        frame_length,
        None, // win_length
        None, // hop_length
        None, // resolution
    );

    let samples_f64: Vec<f64> = samples_i.iter().map(|&s| s as f64).collect();
    let framing = Framing::Center(PadMode::Constant(0.0));
    let fill_unvoiced = f64::NAN;

    let (timestamp, f0, voiced_flag, _voiced_prob) =
        pyin_exec.pyin(&samples_f64, fill_unvoiced, framing);

    let mut pyin_results: Vec<(f64, f32)> = Vec::new();
    for i in 0..f0.len() {
        if voiced_flag[i] && !f0[i].is_nan() && f0[i] > 0.0 {
            pyin_results.push((timestamp[i], f0[i] as f32));
        }
    }
    eprintln!("  pyin-rs: {} voiced frames", pyin_results.len());

    // Write CSVs
    let mut f_ours = File::create("/home/ANT.AMAZON.COM/wssmts/Documents/pitch_ours.csv").unwrap();
    writeln!(f_ours, "time_s,frequency_hz").unwrap();
    for (t, f) in &ours {
        writeln!(f_ours, "{:.4},{:.2}", t, f).unwrap();
    }

    let mut f_pyin =
        File::create("/home/ANT.AMAZON.COM/wssmts/Documents/pitch_pyin_rs.csv").unwrap();
    writeln!(f_pyin, "time_s,frequency_hz").unwrap();
    for (t, f) in &pyin_results {
        writeln!(f_pyin, "{:.4},{:.2}", t, f).unwrap();
    }

    eprintln!("Wrote pitch_ours.csv and pitch_pyin_rs.csv");
}
