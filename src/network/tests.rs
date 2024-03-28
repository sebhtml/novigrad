use more_asserts::assert_gt;
use more_asserts::assert_lt;

use crate::load_dataset;
use crate::train::train_network_on_dataset;
use crate::Dataset;

fn test_network_on_dataset(dataset: &Dataset) {
    let dataset_details = load_dataset(&dataset);
    let training_output = train_network_on_dataset(&dataset_details).expect("Ok");
    assert_gt!(
        training_output.initial_total_error,
        dataset_details.initial_total_error_min
    );
    assert_lt!(
        training_output.final_total_error,
        dataset_details.final_total_error_max
    );
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
    let dataset = Dataset::Simple;
    test_network_on_dataset(&dataset);
}

#[test]
fn mega_man_dataset() {
    let dataset = Dataset::MegaMan;
    test_network_on_dataset(&dataset);
}
