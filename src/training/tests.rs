use more_asserts::assert_ge;
use more_asserts::assert_le;

use crate::load_dataset;
use crate::train_network_on_dataset;
use crate::Dataset;
use crate::Device;

fn test_network_on_dataset(dataset: Dataset, device: &Device) {
    let dataset_details = load_dataset(dataset, device).unwrap();
    let initial_total_error_min = dataset_details.initial_total_error_min;
    let final_total_error_max = dataset_details.final_total_error_max;
    let training_output = train_network_on_dataset(dataset_details).unwrap();
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
    test_network_on_dataset(Dataset::Perceptron, &device);
}

#[test]
fn simple_dataset_cpu() {
    let device = Device::cpu();
    test_network_on_dataset(Dataset::Simple, &device);
}

#[test]
fn simple_dataset_cuda() {
    let device = Device::cuda().unwrap();
    test_network_on_dataset(Dataset::Simple, &device);
}

#[test]
fn mega_man_dataset_cuda() {
    let device = Device::cuda().unwrap();
    test_network_on_dataset(Dataset::MegaMan, &device);
}

#[test]
fn mega_man_attention_dataset_cuda() {
    let device = Device::cuda().unwrap();
    test_network_on_dataset(Dataset::MegaManAttention, &device);
}
