use std::rc::Rc;

use rs_brain::{load_dataset, train_network_on_dataset, Dataset, Device};

fn main() {
    let device = Rc::new(Device::default());
    let dataset = Dataset::Simple;
    let dataset_details = load_dataset(dataset, device.clone());
    train_network_on_dataset(dataset_details, device).expect("Ok");
}
