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

fn main() {

    let host = get_host();

    println!("\nSelected Host Device: {}\n", host.id().name());

    let input_device = host.default_input_device().unwrap();

    let config = input_device
        .default_input_config()
        .expect("Failed to get default input config");
    println!("Default input config: {:?}", config);

}
