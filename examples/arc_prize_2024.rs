use novigrad::{datasets::arc_prize_2024::load_arc_prize_2024, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_arc_prize_2024(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
