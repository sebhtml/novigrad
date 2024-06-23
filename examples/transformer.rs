use novigrad::{train_model, transformer_model::load_transformer_model, Device};

fn main() {
    let device = Device::default();
    let details = load_transformer_model(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
