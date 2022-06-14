use realfft::{RealFftPlanner, RealToComplex};
use splines::{Interpolation, Key, Spline};
use std::sync::Arc;
use textplots::Shape;
use textplots::{Chart, Plot};

const BUFFER_SIZE: usize = 1024;

pub struct SignalDrawer {
    fft: Arc<dyn RealToComplex<f32>>,
    should_clear_screen: bool,
}

impl SignalDrawer {
    pub fn new(should_clear_screen: bool) -> Self {
        Self {
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(BUFFER_SIZE),
            should_clear_screen,
        }
    }

    pub fn draw_data(&self, data: &[f32]) {
        if self.should_clear_screen {
            print!("{}", ansi_escapes::CursorTo::TopLeft);
        }
        self.draw_psd(data);
        self.draw_waveform(data);
    }

    fn draw_waveform(&self, data: &[f32]) {
        println!("Waveform:");
        let (width, height) = get_textplots_window_size();
        let vec = data
            .iter()
            .enumerate()
            .map(|(index, sample)| Key::new(index as f32, *sample as f32, Interpolation::Linear))
            .collect();

        let spline = Spline::from_vec(vec);
        Chart::new_with_y_range(width, height / 2, 0.0, BUFFER_SIZE as f32, -1.0, 1.0)
            .lineplot(&Shape::Continuous(Box::new(|x| {
                spline.sample(x).unwrap_or(0.0)
            })))
            .display();
    }

    fn draw_psd(&self, data: &[f32]) {
        println!("Power Spectral Density:");
        let (width, height) = get_textplots_window_size();

        let mut ff_data = [0.0; BUFFER_SIZE].to_vec();
        ff_data.copy_from_slice(&data[0..BUFFER_SIZE]);
        let mut spectrum = self.fft.make_output_vec();
        self.fft.process(&mut ff_data, &mut spectrum).unwrap();

        let vec: Vec<_> = spectrum
            .into_iter()
            .map(|complex| complex.norm_sqr())
            .enumerate()
            .map(|(index, val)| ((index as f32).log2() * 115.4, val))
            .map(|(index, sample)| Key::new(index as f32, sample as f32, Interpolation::Cosine))
            .collect();

        let spline = Spline::from_vec(vec);

        Chart::new_with_y_range(width, height / 2, 0.0, BUFFER_SIZE as f32, 0.0, 50.0)
            .lineplot(&Shape::Continuous(Box::new(|x| {
                spline.sample(x).unwrap_or(0.0)
            })))
            .display();
    }
}

fn get_textplots_window_size() -> (u32, u32) {
    let (mut width, height) = termion::terminal_size().unwrap();
    width = width * 2 - 11;
    let height = height as f32 * 1.6;
    (width as u32, height as u32 - 10)
}