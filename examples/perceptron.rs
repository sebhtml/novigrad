use rs_brain::{load_perceptron, train_network_on_dataset, Device};

fn main() {
    let device = Device::cuda().unwrap();
    let dataset_details = load_perceptron(&device).unwrap();
    train_network_on_dataset(dataset_details).unwrap();
}
