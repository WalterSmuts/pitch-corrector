//! Spectrogram render throughput: `cargo run --release --example bench_spectrogram`
use pitch_corrector::session::SpectrogramRenderer;

fn main() {
    let audio: Vec<f32> = (0..48000 * 3)
        .map(|i| {
            let t = i as f32 / 48000.0;
            let f = 120.0 + 400.0 * t;
            let mut s = (std::f32::consts::TAU * f * t).sin() * 0.4;
            s += (std::f32::consts::TAU * 2.0 * f * t).sin() * 0.2;
            s += ((i as u32).wrapping_mul(2654435761) >> 16) as f32 / 65536.0 * 0.05;
            s
        })
        .collect();
    let mut r = SpectrogramRenderer::new();
    let mut rgba = Vec::new();
    r.render(&audio, 0.0, 140.0, 8, 340, 0.0, 1.0, &mut rgba); // warm planner
    for (label, height) in [("fft-only (h=1)", 1usize), ("full (h=340)", 340)] {
        let (width, reps) = (1024usize, 20usize);
        let t0 = std::time::Instant::now();
        for i in 0..reps {
            r.render(
                &audio,
                (i * 100) as f64,
                140.0,
                width,
                height,
                0.0,
                1.0,
                &mut rgba,
            );
        }
        let dt = t0.elapsed().as_secs_f64();
        let cols = (width * reps) as f64 / dt;
        println!(
            "{label}: {cols:.0} cols/s | 128-col slice = {:.2} ms",
            128.0 / cols * 1000.0
        );
    }
    let (width, height, reps) = (1024usize, 340usize, 20usize);
    let t0 = std::time::Instant::now();
    for i in 0..reps {
        r.render(
            &audio,
            (i * 100) as f64,
            140.0,
            width,
            height,
            0.0,
            1.0,
            &mut rgba,
        );
    }
    let dt = t0.elapsed().as_secs_f64();
    let cols = (width * reps) as f64 / dt;
    println!(
        "{cols:.0} cols/s | 128-col budget slice = {:.2} ms",
        128.0 / cols * 1000.0
    );
}
