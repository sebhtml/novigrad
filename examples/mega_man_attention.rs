use rs_brain::{load_dataset, train_network_on_dataset, Dataset, Device};

fn main() {
    let device = Device::cuda().unwrap();
    let dataset = Dataset::MegaManAttention;
    let dataset_details = load_dataset(dataset, &device).unwrap();
    train_network_on_dataset(dataset_details).unwrap();
}