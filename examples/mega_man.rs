use novigrad::{mega_man::load_mega_man_model, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_mega_man_model(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
