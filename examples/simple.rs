use rs_brain::{load_model_details, train_model, Device, ModelEnum};

fn main() {
    let device = Device::cuda().unwrap();
    let model = ModelEnum::Simple;
    let details = load_model_details(model, &device).unwrap();
    train_model(details).unwrap();
}
