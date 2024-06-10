use novigrad::{mega_man_attention::load_mega_man_attention_model, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_mega_man_attention_model(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
