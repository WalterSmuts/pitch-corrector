pub trait Interpolate<const METHOD: InterpolationMethod> {
    fn interpolate_sample(&self, index: f32) -> f32;

    fn interpolate_samples(&self, buffer: &mut [f32]) {
        buffer.iter_mut().for_each(|sample| {
            *sample = self.interpolate_sample(*sample);
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum InterpolationMethod {
    Linear,
    WhittakerShannon,
}

impl Interpolate<{ InterpolationMethod::Linear }> for [f32] {
    fn interpolate_sample(&self, index: f32) -> f32 {
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            return self[lower_index];
        }

        let lower = self[lower_index];
        let upper = self[upper_index];

        upper * (index - lower_index as f32) + lower * (upper_index as f32 - index)
    }
}

// TODO: Investigate fft optimization
impl Interpolate<{ InterpolationMethod::WhittakerShannon }> for [f32] {
    fn interpolate_sample(&self, j: f32) -> f32 {
        let mut sum = 0.0;
        for (i, x) in self.iter().enumerate() {
            sum += x * sinc(j - i as f32);
        }
        sum
    }
}

fn sinc(x: f32) -> f32 {
    use std::f32::consts::PI;
    if x == 0.0 {
        1.0
    } else {
        (x * PI).sin() / (x * PI)
    }
}

#[cfg(test)]
mod test {
    use super::Interpolate;
    use super::InterpolationMethod;
    use std::f32::consts::TAU;

    const BUFFER_SIZE: usize = 1024;
    const TEST_SAMPLE_SIZE: usize = BUFFER_SIZE * 10;
    const TEST_EQUALITY_EPISLON: f32 = 0.002;

    #[test]
    fn linear_interpolate() {
        // Define linear signal
        let signal = |x: f32| 2.0 * x;

        // Sample initial buffer
        let mut buffer = [0.0; BUFFER_SIZE];
        for (x, y) in buffer.iter_mut().enumerate() {
            *y = signal(x as f32);
        }

        // Interpolate at random samples and assert signal is equal to interpolation
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>() * (BUFFER_SIZE as f32 - 1.0);
            assert_eq!(
                signal(x),
                buffer.interpolate_sample(x, InterpolationMethod::Linear)
            )
        }
    }

    #[test]
    fn whittaker_shannon_interpolate() {
        // Define band-limited signal
        let signal = |x: f32| (TAU * x / BUFFER_SIZE as f32).sin();

        // Sample initial buffer
        let mut buffer = [0.0; BUFFER_SIZE];
        for (x, y) in buffer.iter_mut().enumerate() {
            *y = signal(x as f32);
        }

        // Interpolate at random samples and assert signal is equal to interpolation
        for _ in 0..TEST_SAMPLE_SIZE {
            let x = rand::random::<f32>() * BUFFER_SIZE as f32;
            approx::assert_abs_diff_eq!(
                signal(x),
                buffer.interpolate_sample(x, InterpolationMethod::WhittakerShannon),
                epsilon = TEST_EQUALITY_EPISLON
            )
        }
    }
}
