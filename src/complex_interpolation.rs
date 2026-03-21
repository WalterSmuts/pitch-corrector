use easyfft::num_complex::Complex;

pub trait ComplexInterpolate {
    fn interpolate_sample(&self, index: f32) -> Complex<f32>;
    #[allow(dead_code)]
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
        let t = index - lower_index as f32;

        // Interpolate magnitude linearly
        let mag = lower.norm() * (1.0 - t) + upper.norm() * t;

        // Interpolate phase with wrapping awareness
        let mut phase_diff = upper.arg() - lower.arg();
        if phase_diff > std::f32::consts::PI {
            phase_diff -= std::f32::consts::TAU;
        } else if phase_diff < -std::f32::consts::PI {
            phase_diff += std::f32::consts::TAU;
        }
        let phase = lower.arg() + phase_diff * t;

        Complex::from_polar(mag, phase)
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

    #[test]
    fn phase_wrapping_interpolation_takes_short_arc() {
        // Two bins with phases near ±π: should interpolate through ±π, not through 0
        let a = Complex::from_polar(1.0, std::f32::consts::PI - 0.1);
        let b = Complex::from_polar(1.0, -std::f32::consts::PI + 0.1);
        let array = [a, b];

        let mid = array.interpolate_sample(0.5);
        // Short arc midpoint should be near ±π, not near 0
        assert!(
            mid.arg().abs() > 2.0,
            "Phase should be near ±π but was {}",
            mid.arg()
        );
    }
}
