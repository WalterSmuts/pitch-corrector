pub trait Interpolate {
    fn interpolate_sample(&self, index: f32) -> f32;
    fn interpolate_samples(&self, buffer: &mut [f32]);
}

impl Interpolate for [f32] {
    fn interpolate_sample(&self, index: f32) -> f32 {
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        let lower = self[lower_index];
        let upper = self[upper_index];

        (index - lower_index as f32) * lower + (upper_index as f32 - index) * upper
    }

    fn interpolate_samples(&self, buffer: &mut [f32]) {
        buffer.iter_mut().for_each(|sample| {
            *sample = self.interpolate_sample(*sample);
        })
    }
}
