use novigrad::{datasets::simple::load_simple_dataset, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_simple_dataset(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
