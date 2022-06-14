use crate::StreamProcessor;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample;
use cpal::{BufferSize, Stream};
use cpal::{InputCallbackInfo, OutputCallbackInfo};
use cpal::{SampleRate, StreamConfig};
use std::sync::Arc;

const SAMPLE_RATE: u32 = 44100;
const BUFFER_SIZE: usize = 128;

pub fn setup_passthrough_processor<T: 'static>(processor: T) -> (Stream, Stream)
where
    T: StreamProcessor + Send + Sync,
{
    let input_passthrough_processor = Arc::new(processor);
    let output_passthrough_processor = input_passthrough_processor.clone();

    let input_stream = get_input_stream(move |data: &[f32], _| {
        for datum in data {
            input_passthrough_processor.push_sample(*datum);
        }
    });
    input_stream.play().unwrap();

    let output_stream = get_output_stream(move |data: &mut [f32], _| {
        for sample in data.iter_mut() {
            *sample = Sample::from(&output_passthrough_processor.pop_sample().unwrap_or(0.0));
        }
    });
    output_stream.play().unwrap();
    (input_stream, output_stream)
}

fn get_input_stream<T, D>(handler: D) -> cpal::Stream
where
    T: Sample,
    D: FnMut(&[T], &InputCallbackInfo) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");

    device
        .build_input_stream(
            &StreamConfig {
                channels: 1,
                sample_rate: SampleRate(SAMPLE_RATE),
                buffer_size: BufferSize::Fixed((BUFFER_SIZE) as u32),
            },
            handler,
            |_| panic!("Error from ALSA on input"),
        )
        .unwrap()
}

fn get_output_stream<T, D>(handler: D) -> cpal::Stream
where
    T: Sample,
    D: FnMut(&mut [T], &OutputCallbackInfo) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No input device available");

    device
        .build_output_stream(
            &StreamConfig {
                channels: 1,
                sample_rate: SampleRate(SAMPLE_RATE),
                buffer_size: BufferSize::Fixed((BUFFER_SIZE) as u32),
            },
            handler,
            |_| panic!("Error from ALSA on output"),
        )
        .unwrap()
}
