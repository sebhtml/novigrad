use novigrad::{datasets::simple::load_simple, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_simple(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
