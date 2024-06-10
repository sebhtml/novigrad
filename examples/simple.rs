use novigrad::{simple::load_simple_model, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_simple_model(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
