use novigrad::{datasets::addition::load_addition_dataset, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_addition_dataset(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
