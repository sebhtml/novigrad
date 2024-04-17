use rs_brain::{load_dataset, train_network_on_dataset, Dataset};

fn main() {
    let dataset = Dataset::MegaMan;
    let dataset_details = load_dataset(&dataset);
    train_network_on_dataset(dataset_details).expect("Ok");
}
