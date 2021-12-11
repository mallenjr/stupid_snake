use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::BufWriter;

fn get_host() -> cpal::Host {
    let default_host_id = &cpal::default_host().id();

    let host = if cfg!(target_os = "linux") {
        let host_id = *cpal::available_hosts()
            .iter()
            .find(|h| h.name()
            .contains("jack"))
            .unwrap_or(default_host_id);

        cpal::host_from_id(host_id).unwrap()
    } else if cfg!(target_os = "windows") {
        let host_id = *cpal::available_hosts()
            .iter()
            .find(|h| h.name()
            .contains("asio"))
            .unwrap_or(default_host_id);
        
        cpal::host_from_id(host_id).unwrap()
    } else {
        cpal::default_host()
    };

    host
}

fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    match format {
        cpal::SampleFormat::U16 => hound::SampleFormat::Int,
        cpal::SampleFormat::I16 => hound::SampleFormat::Int,
        cpal::SampleFormat::F32 => hound::SampleFormat::Float,
    }
}

fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate().0 as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}

type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<BufWriter<File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: cpal::Sample,
    U: cpal::Sample + hound::Sample,
{
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = cpal::Sample::from(&sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}


fn main() -> Result<(), anyhow::Error> {

    let host = get_host();

    println!("\nSelected Host Device: {}\n", host.id().name());

    let input_device = host.default_input_device().unwrap();

    let config = input_device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("Default input config: {:?}", config);

    // The WAV file we're recording to.
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recorded.wav");
    let spec = wav_spec_from_config(&config);
    let writer = hound::WavWriter::create(PATH, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let writer_2 = writer.clone();

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => input_device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<f32, f32>(data, &writer_2),
            err_fn,
        )?,
        cpal::SampleFormat::I16 => input_device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<i16, i16>(data, &writer_2),
            err_fn,
        )?,
        cpal::SampleFormat::U16 => input_device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<u16, i16>(data, &writer_2),
            err_fn,
        )?,
    };

    // Let recording go for roughly three seconds.
    std::thread::sleep(std::time::Duration::from_secs(3));
    drop(stream);
    writer.lock().unwrap().take().unwrap().finalize()?;
    println!("Recording {} complete!", PATH);
    Ok(())
}
