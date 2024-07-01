use novigrad::{
    datasets::mega_man_attention_head::load_mega_man_attention_head_dataset, train_model, Device,
};

fn main() {
    let device = Device::default();
    let details = load_mega_man_attention_head_dataset(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
