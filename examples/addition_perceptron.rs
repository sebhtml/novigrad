use novigrad::{datasets::addition_perceptron::load_addition_perceptron, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_addition_perceptron(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
