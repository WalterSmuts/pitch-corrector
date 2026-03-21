use crate::signal_processing::StreamProcessor;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use cpal::{BufferSize, Stream};
use cpal::{InputCallbackInfo, OutputCallbackInfo};
use cpal::{Sample, SizedSample};
use log::{debug, error, info};
use std::sync::Arc;

const SAMPLE_RATE: u32 = 44100;
const BUFFER_SIZE: usize = 1024;

pub fn setup_passthrough_processor<T>(processor: T) -> (Stream, Stream)
where
    T: StreamProcessor + Send + Sync + 'static,
{
    info!("Setting up hardware");

    let input_passthrough_processor = Arc::new(processor);
    let output_passthrough_processor = input_passthrough_processor.clone();

    let input_stream = get_input_stream(move |data: &[f32], _| {
        for datum in data {
            input_passthrough_processor.push_sample(*datum);
        }
    });

    match input_stream.play() {
        Ok(_) => info!("Input stream started successfully"),
        Err(e) => error!("Failed to start input stream: {}", e),
    }

    let output_stream = get_output_stream(move |data: &mut [f32], _| {
        for sample in data.iter_mut() {
            *sample = output_passthrough_processor.pop_sample().unwrap_or(0.0);
        }
    });

    match output_stream.play() {
        Ok(_) => info!("Output stream started successfully"),
        Err(e) => error!("Failed to start output stream: {}", e),
    }

    (input_stream, output_stream)
}

fn get_input_stream<T, D>(handler: D) -> cpal::Stream
where
    T: Sample + SizedSample,
    D: FnMut(&[T], &InputCallbackInfo) + Send + 'static,
{
    debug!("Getting input stream");
    let host = cpal::default_host();
    info!("Using audio host: {}", host.id().name());

    let device = host
        .default_input_device()
        .expect("No input device available");
    info!(
        "Using input device: {}",
        device
            .description()
            .map_or("Unknown".into(), |d| d.to_string())
    );

    let config = StreamConfig {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        buffer_size: BufferSize::Fixed(BUFFER_SIZE as u32),
    };
    debug!("Input stream config: {:?}", config);

    device
        .build_input_stream(
            &config,
            handler,
            |err| error!("Input stream error: {}", err),
            None,
        )
        .expect("Failed to build input stream")
}

pub fn get_output_stream<T, D>(handler: D) -> cpal::Stream
where
    T: Sample + SizedSample,
    D: FnMut(&mut [T], &OutputCallbackInfo) + Send + 'static,
{
    debug!("Getting output stream");
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device available");
    info!(
        "Using output device: {}",
        device
            .description()
            .map_or("Unknown".into(), |d| d.to_string())
    );

    let config = StreamConfig {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        buffer_size: BufferSize::Fixed(BUFFER_SIZE as u32),
    };
    debug!("Output stream config: {:?}", config);

    device
        .build_output_stream(
            &config,
            handler,
            |err| error!("Output stream error: {}", err),
            None,
        )
        .expect("Failed to build output stream")
}
