use more_asserts::assert_ge;
use more_asserts::assert_le;

use crate::datasets::addition::load_addition_dataset;
use crate::datasets::geoffroy_hinton::load_geoffroy_hinton_dataset;
use crate::datasets::mega_man_linear::load_mega_man_linear_dataset;
use crate::datasets::mega_man_multi_head_attention::load_mega_man_attention_dataset;
use crate::datasets::simple::load_simple_dataset;
use crate::datasets::DatasetDetails;
use crate::display::TensorPrinter;
use crate::train_model;
use crate::BinaryOperator;
use crate::Device;
use crate::OptimizerTrait;
use crate::UnaryModel;

fn test_model(
    details: DatasetDetails<
        impl UnaryModel,
        impl BinaryOperator,
        impl OptimizerTrait,
        impl TensorPrinter,
    >,
) {
    let expected_initial_total_loss_min = details.initial_metrics.total_loss;
    let expected_final_total_loss_max = details.final_metrics.total_loss;
    let expected_initial_total_perplexity_min = details.initial_metrics.total_next_token_perplexity;
    let expected_final_total_perplexity_max = details.final_metrics.total_next_token_perplexity;
    let maximum_incorrect_argmaxes = details.maximum_incorrect_argmaxes;
    let training_output = train_model::<f32>(details).unwrap();

    // Verify total loss
    assert_le!(
        expected_initial_total_loss_min,
        training_output.initial_metrics.total_loss
    );
    assert_ge!(
        expected_final_total_loss_max,
        training_output.final_metrics.total_loss
    );

    // Verify total perplexity
    if !expected_initial_total_perplexity_min.is_nan() {
        assert_le!(
            expected_initial_total_perplexity_min,
            training_output.initial_metrics.total_next_token_perplexity
        );
    }
    assert_ge!(
        expected_final_total_perplexity_max,
        training_output.final_metrics.total_next_token_perplexity
    );

    let mut incorrect_argmaxes = 0;
    // Verify argmaxes
    for i in 0..training_output.expected_argmax_values.len() {
        let expected = training_output.actual_argmax_values[i];
        let actual = training_output.expected_argmax_values[i];
        if expected != actual {
            incorrect_argmaxes += 1;
        }
    }
    assert_le!(incorrect_argmaxes, maximum_incorrect_argmaxes);
}

#[test]
fn addition_with_perceptron() {
    let device = Device::default();
    let details = load_addition_dataset(&device).unwrap();
    test_model(details);
}

#[test]
fn simple() {
    let device = Device::default();
    let details = load_simple_dataset(&device).unwrap();
    test_model(details);
}

#[test]
fn mega_man_with_linear() {
    let device = Device::default();
    let details = load_mega_man_linear_dataset(&device).unwrap();
    test_model(details);
}

#[test]
fn mega_man_with_attention() {
    let device = Device::default();
    let details = load_mega_man_attention_dataset(&device).unwrap();
    test_model(details);
}

// This test is currently ignored because it'S too slow.
#[ignore]
#[test]
fn geoffroy_hinton_with_transformer() {
    let device = Device::default();
    let details = load_geoffroy_hinton_dataset(&device).unwrap();
    test_model(details);
}
