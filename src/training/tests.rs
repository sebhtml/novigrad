use more_asserts::assert_ge;
use more_asserts::assert_le;

use crate::load_model_details;
use crate::train_model;
use crate::Device;
use crate::ModelEnum;

fn test_model(model: ModelEnum, device: &Device) {
    let details = load_model_details(model, device).unwrap();
    let initial_total_error_min = details.initial_total_error_min;
    let final_total_error_max = details.final_total_error_max;
    let training_output = train_model::<f32>(details).unwrap();
    assert_ge!(training_output.initial_total_error, initial_total_error_min);
    assert_le!(training_output.final_total_error, final_total_error_max);
    for i in 0..training_output.expected_argmax_values.len() {
        let expected = training_output.actual_argmax_values[i];
        let actual = training_output.expected_argmax_values[i];
        assert_eq!(
            expected, actual,
            "example: {}, expected_argmax: {}, actual_argmax: {}",
            i, expected, actual,
        );
    }
}

#[test]
fn perceptron_model_cpu() {
    let device = Device::cpu();
    test_model(ModelEnum::Perceptron, &device);
}

#[test]
fn simple_model_cpu() {
    let device = Device::cpu();
    test_model(ModelEnum::Simple, &device);
}

#[test]
fn simple_model_cuda() {
    let device = Device::cuda().unwrap();
    test_model(ModelEnum::Simple, &device);
}

#[test]
fn mega_man_model_cpu() {
    let device = Device::cpu();
    test_model(ModelEnum::MegaMan, &device);
}

#[test]
fn mega_man_model_cuda() {
    let device = Device::cuda().unwrap();
    test_model(ModelEnum::MegaMan, &device);
}

#[test]
fn mega_man_attention_model_cpu() {
    let device = Device::cpu();
    test_model(ModelEnum::MegaManAttention, &device);
}

#[test]
fn mega_man_attention_model_cuda() {
    let device = Device::cuda().unwrap();
    test_model(ModelEnum::MegaManAttention, &device);
}
