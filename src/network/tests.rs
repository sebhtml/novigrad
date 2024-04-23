use std::rc::Rc;

use more_asserts::assert_gt;
use more_asserts::assert_lt;

use crate::load_dataset;
use crate::train_network_on_dataset;
use crate::Dataset;
use crate::Device;

fn test_network_on_dataset(dataset: Dataset, device: Device) {
    let device = Rc::new(device);
    let dataset_details = load_dataset(dataset, device.clone());
    let initial_total_error_min = dataset_details.initial_total_error_min;
    let final_total_error_max = dataset_details.final_total_error_max;
    let training_output = train_network_on_dataset(dataset_details, device).expect("Ok");
    assert_gt!(training_output.initial_total_error, initial_total_error_min);
    assert_lt!(training_output.final_total_error, final_total_error_max);
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
fn simple_dataset() {
    let device = Device::default();
    let dataset = Dataset::Simple;
    test_network_on_dataset(dataset, device);
}

#[test]
fn mega_man_dataset() {
    let device = Device::default();
    let dataset = Dataset::MegaMan;
    test_network_on_dataset(dataset, device);
}
