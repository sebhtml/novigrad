use more_asserts::assert_gt;
use more_asserts::assert_lt;

use crate::train::train_network_on_dataset;
use crate::Dataset;

#[test]
fn simple_dataset() {
    let dataset = Dataset::Simple;
    let test_output: crate::train::NetworkTestOutput = train_network_on_dataset(&dataset);
    assert_gt!(test_output.initial_total_error, 0.5);
    assert_lt!(test_output.final_total_error, 0.000001);
}

#[test]
fn mega_man_dataset() {
    let dataset = Dataset::MegaMan;
    let test_output = train_network_on_dataset(&dataset);
    assert_gt!(test_output.initial_total_error, 0.4);
    assert_lt!(test_output.final_total_error, 0.000001);
}
