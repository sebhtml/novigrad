use more_asserts::assert_ge;
use more_asserts::assert_le;

use crate::geoffroy_hinton_transformer_model::load_geoffroy_hinton_transformer_model;
use crate::mega_man::load_mega_man_model;
use crate::mega_man_attention::load_mega_man_attention_model;
use crate::perceptron::load_perceptron;
use crate::simple::load_simple_model;
use crate::train_model;
use crate::BinaryOperator;
use crate::Device;
use crate::ModelDetails;
use crate::OptimizerTrait;
use crate::UnaryModel;

fn test_model(details: ModelDetails<impl UnaryModel, impl BinaryOperator, impl OptimizerTrait>) {
    let expected_initial_total_loss_min = details.initial_metrics.total_loss;
    let expected_final_total_loss_max = details.final_metrics.total_loss;
    let expected_initial_total_perplexity_min = details.initial_metrics.total_perplexity;
    let expected_final_total_perplexity_max = details.final_metrics.total_perplexity;
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
            training_output.initial_metrics.total_perplexity
        );
    }
    assert_ge!(
        expected_final_total_perplexity_max,
        training_output.final_metrics.total_perplexity
    );

    // Verify argmaxes
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
fn perceptron_model() {
    let device = Device::default();
    let details = load_perceptron(&device).unwrap();
    test_model(details);
}

#[test]
fn simple_model() {
    let device = Device::default();
    let details = load_simple_model(&device).unwrap();
    test_model(details);
}

#[test]
fn mega_man_model() {
    let device = Device::default();
    let details = load_mega_man_model(&device).unwrap();
    test_model(details);
}

#[test]
fn mega_man_attention_model() {
    let device = Device::default();
    let details = load_mega_man_attention_model(&device).unwrap();
    test_model(details);
}

#[test]
fn geoffroy_hinton_transformer_model() {
    let device = Device::default();
    let details = load_geoffroy_hinton_transformer_model(&device).unwrap();
    test_model(details);
}
