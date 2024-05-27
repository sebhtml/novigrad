use more_asserts::assert_ge;
use more_asserts::assert_le;

use crate::load_model_details;
use crate::train_model;
use crate::Device;
use crate::ModelEnum;

fn test_model(model: ModelEnum, device: &Device) {
    let details = load_model_details(model, device).unwrap();
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
    assert_le!(
        expected_initial_total_perplexity_min,
        training_output.initial_metrics.total_perplexity
    );
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
    test_model(ModelEnum::Perceptron, &device);
}

#[test]
fn simple_model() {
    let device = Device::default();
    test_model(ModelEnum::Simple, &device);
}

#[test]
fn mega_man_model() {
    let device = Device::default();
    test_model(ModelEnum::MegaMan, &device);
}

#[test]
fn mega_man_attention_model() {
    let device = Device::default();
    test_model(ModelEnum::MegaManAttention, &device);
}
