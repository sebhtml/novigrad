use novigrad::{datasets::arc_prize_2024::load_arc_dataset, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_arc_dataset(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
