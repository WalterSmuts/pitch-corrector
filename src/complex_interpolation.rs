use realfft::num_complex::Complex;

pub trait ComplexInterpolate {
    fn interpolate_sample(&self, index: f32) -> Complex<f32>;
    fn interpolate_samples(&self, buffer: &[f32]) -> Vec<Complex<f32>>;
}

impl ComplexInterpolate for [Complex<f32>] {
    fn interpolate_sample(&self, index: f32) -> Complex<f32> {
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            return self[lower_index];
        }

        let lower = self[lower_index];
        let upper = self[upper_index];

        Complex::from_polar(
            lower.norm() * (upper_index as f32 - index)
                + upper.norm() * (index - lower_index as f32),
            lower.arg() * (upper_index as f32 - index) + upper.arg() * (index - lower_index as f32),
        )
    }

    fn interpolate_samples(&self, buffer: &[f32]) -> Vec<Complex<f32>> {
        let mut output = Vec::with_capacity(buffer.len());
        for index in buffer {
            output.push(self.interpolate_sample(*index));
        }
        output
    }
}

#[cfg(test)]
mod test {
    use super::*;
    const TEST_EQUALITY_EPISLON: f32 = 0.1;

    #[test]
    fn test_complex_interpolation_halfway() {
        let a: Complex<f32> = Complex::from_polar(1.0, 1.0);
        let b = Complex::from_polar(3.0, 2.0);
        let array = [a, b];
        assert_eq!(array.interpolate_sample(0.5).norm(), 2.0);
        assert_eq!(array.interpolate_sample(0.5).arg(), 1.5);
    }

    #[test]
    fn test_complex_interpolation_close_to_sample() {
        let a_norm = 1.0;
        let a_arg = 1.0;
        let a: Complex<f32> = Complex::from_polar(a_norm, a_arg);
        let b = Complex::from_polar(3.0, 2.0);
        let array = [a, b];

        approx::assert_abs_diff_eq!(
            array.interpolate_sample(0.001).norm(),
            a_norm,
            epsilon = TEST_EQUALITY_EPISLON
        );

        approx::assert_abs_diff_eq!(
            array.interpolate_sample(0.001).arg(),
            a_arg,
            epsilon = TEST_EQUALITY_EPISLON
        );
    }
}
