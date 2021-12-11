use cpal::traits::{DeviceTrait, HostTrait};

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

fn get_input_device(host: cpal::Host) -> cpal::Device {
    let input_devices = host.input_devices().expect("Failed to query input devices");

    for (pos, device) in input_devices.enumerate() {
        let configs = device.supported_input_configs();

        if let Err(_e) = configs { // We could not query inputs from this device
            continue
        }

        println!("Device {}: {}", pos, device.name().expect("Could not display device name"))
    }

    host.default_input_device().unwrap()
}

fn main() {

    let host = get_host();

    println!("\nSelected Host Device: {}\n", host.id().name());

    let _input_device = get_input_device(host);

}
