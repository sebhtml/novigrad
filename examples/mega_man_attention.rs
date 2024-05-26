use novigrad::{load_model_details, train_model, Device, ModelEnum};

fn main() {
    let device = Device::default();
    let model = ModelEnum::MegaManAttention;
    let details = load_model_details(model, &device).unwrap();
    train_model::<f32>(details).unwrap();
}
