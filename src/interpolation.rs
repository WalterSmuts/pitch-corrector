pub trait Interpolate {
    fn interpolate_sample(&self, index: f32, method: InterpolationMethod) -> f32;
    fn interpolate_samples(&self, buffer: &mut [f32], method: InterpolationMethod);
}

trait LinearInterpolate {
    fn interpolate_sample(&self, index: f32) -> f32;
}

trait WhittakerShannonInterpolate {
    fn interpolate_sample(&self, index: f32) -> f32;
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub enum InterpolationMethod {
    Linear,
    WhittakerShannon,
}

impl Interpolate for [f32] {
    fn interpolate_sample(&self, index: f32, method: InterpolationMethod) -> f32 {
        match method {
            InterpolationMethod::Linear => LinearInterpolate::interpolate_sample(self, index),
            InterpolationMethod::WhittakerShannon => {
                todo!()
            }
        }
    }

    fn interpolate_samples(&self, buffer: &mut [f32], method: InterpolationMethod) {
        buffer.iter_mut().for_each(|sample| {
            *sample = Interpolate::interpolate_sample(self, *sample, method);
        })
    }
}

impl LinearInterpolate for [f32] {
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

#[cfg(test)]
mod test {
    use super::Interpolate;
    use super::InterpolationMethod;

    const BUFFER_SIZE: usize = 1024;
    const TEST_SAMPLE_SIZE: usize = BUFFER_SIZE * 10;

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
}
