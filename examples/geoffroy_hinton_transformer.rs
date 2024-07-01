use novigrad::{
    datasets::geoffroy_hinton_transformer::load_geoffroy_hinton_transformer, train_model, Device,
};

fn main() {
    let device = Device::default();
    let details = load_geoffroy_hinton_transformer(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
