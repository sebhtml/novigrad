use novigrad::{
    geoffroy_hinton_transformer_model::load_geoffroy_hinton_transformer_model, train_model, Device,
};

fn main() {
    let device = Device::default();
    let details = load_geoffroy_hinton_transformer_model(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
