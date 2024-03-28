use more_asserts::assert_gt;
use more_asserts::assert_lt;

use crate::train::train_network_on_dataset;
use crate::Dataset;

#[test]
fn simple_dataset() {
    let dataset = Dataset::Simple;
    let test_output = train_network_on_dataset(&dataset).expect("Ok");
    assert_gt!(test_output.initial_total_error, 2.0);
    assert_lt!(test_output.final_total_error, 0.00025);
}

#[test]
fn mega_man_dataset() {
    let dataset = Dataset::MegaMan;
    let test_output = train_network_on_dataset(&dataset).expect("Ok");
    assert_gt!(test_output.initial_total_error, 50.0);
    assert_lt!(test_output.final_total_error, 0.002);
}
