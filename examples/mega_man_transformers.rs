use novigrad::{
    datasets::mega_man_transformers::load_mega_man_transformers, train_model, Device
};

fn main() {
    let device = Device::default();
    let details = load_mega_man_transformers(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
