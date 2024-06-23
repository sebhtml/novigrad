use novigrad::{multi_head_attention_model::load_multi_head_attention_model, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_multi_head_attention_model(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
