use novigrad::{
    datasets::colored_mosaic_puzzles::load_colored_mosaic_puzzles, train_model, Device,
};

fn main() {
    let device = Device::default();
    let details = load_colored_mosaic_puzzles(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
