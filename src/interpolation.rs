pub trait Interpolater {
    fn interpolate_sample(&self, index: f32) -> f32;

    fn interpolate_samples(&self, buffer: &mut [f32]) {
        buffer.iter_mut().for_each(|sample| {
            *sample = self.interpolate_sample(*sample);
        })
    }
}

pub struct LinearInterpolater<'a> {
    inner: &'a [f32],
}

impl<'a> LinearInterpolater<'a> {
    pub fn new(inner: &'a [f32]) -> Self {
        Self { inner }
    }
}

impl<'a> Interpolater for LinearInterpolater<'a> {
    fn interpolate_sample(&self, index: f32) -> f32 {
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            return self.inner[lower_index];
        }

        let lower = self.inner[lower_index];
        let upper = self.inner[upper_index];

        upper * (index - lower_index as f32) + lower * (upper_index as f32 - index)
    }
}

struct WhittakerShannonInterpolator<'a> {
    inner: &'a [f32],
}

impl<'a> WhittakerShannonInterpolator<'a> {
    pub fn new(inner: &'a [f32]) -> Self {
        Self { inner }
    }
}

// TODO: Investigate fft optimization
impl<'a> Interpolater for WhittakerShannonInterpolator<'a> {
    fn interpolate_sample(&self, j: f32) -> f32 {
        let mut sum = 0.0;
        for (i, x) in self.inner.iter().enumerate() {
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
    use super::*;
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
                LinearInterpolater::new(&buffer).interpolate_sample(x)
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
                WhittakerShannonInterpolator::new(&buffer).interpolate_sample(x),
                epsilon = TEST_EQUALITY_EPISLON
            )
        }
    }
}
