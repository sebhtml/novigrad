use more_asserts::assert_gt;
use more_asserts::assert_lt;

use crate::train::train_network_on_dataset;
use crate::Dataset;

#[test]
fn simple_dataset() {
    let dataset = Dataset::Simple;
    let training_output = train_network_on_dataset(&dataset).expect("Ok");
    assert_gt!(training_output.initial_total_error, 2.0);
    assert_lt!(training_output.final_total_error, 0.00025);
}

#[test]
fn mega_man_dataset() {
    let dataset = Dataset::MegaMan;
    let training_output = train_network_on_dataset(&dataset).expect("Ok");
    assert_gt!(training_output.initial_total_error, 50.0);
    assert_lt!(training_output.final_total_error, 0.002);
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
