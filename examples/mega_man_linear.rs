use novigrad::{datasets::mega_man_linear::load_mega_man_linear, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_mega_man_linear(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
