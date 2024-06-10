use novigrad::{perceptron::load_perceptron, train_model, Device};

fn main() {
    let device = Device::default();
    let details = load_perceptron(&device).unwrap();
    train_model::<f32>(details).unwrap();
}
