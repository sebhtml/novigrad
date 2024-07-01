use novigrad::{
    datasets::mega_man_attention_head::load_mega_man_attention_head, train_model, Device,
};

fn main() {
    let device = Device::default();
    let details = load_mega_man_attention_head(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
