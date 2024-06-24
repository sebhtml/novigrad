use novigrad::{
    datasets::mega_man_multi_head_attention::load_mega_man_attention_dataset, train_model, Device,
};

fn main() {
    let device = Device::default();
    let details = load_mega_man_attention_dataset(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
