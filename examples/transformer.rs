use novigrad::{datasets::geoffroy_hinton::load_geoffroy_hinton_dataset, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_geoffroy_hinton_dataset(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
