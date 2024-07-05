use more_asserts::assert_ge;
use more_asserts::assert_le;

use crate::datasets::addition_perceptron::load_addition_perceptron;
use crate::datasets::geoffroy_hinton_transformer::load_geoffroy_hinton_transformer;
use crate::datasets::mega_man_attention_head::load_mega_man_attention_head;
use crate::datasets::mega_man_linear::load_mega_man_linear;
use crate::datasets::mega_man_multi_head_attention::load_mega_man_multi_head_attention;
use crate::datasets::simple::load_simple;
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
    let expected_initial_total_loss_min = details.initial_metrics_min.total_loss;
    let expected_final_total_loss_max = details.final_metrics_max.total_loss;
    let maximum_incorrect_argmaxes = details.maximum_incorrect_predicted_next_tokens;
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
fn addition_perceptron() {
    let device = Device::default();
    let details = load_addition_perceptron(&device).unwrap();
    test_model(details);
}

#[test]
fn simple() {
    let device = Device::default();
    let details = load_simple(&device).unwrap();
    test_model(details);
}

#[test]
fn mega_man_linear() {
    let device = Device::default();
    let details = load_mega_man_linear(&device).unwrap();
    test_model(details);
}

#[test]
fn mega_man_attention_head() {
    let device = Device::default();
    let details = load_mega_man_attention_head(&device).unwrap();
    test_model(details);
}

#[test]
fn mega_man_multi_head_attention() {
    let device = Device::default();
    let details = load_mega_man_multi_head_attention(&device).unwrap();
    test_model(details);
}

#[test]
fn geoffroy_hinton_transformer() {
    let device = Device::default();
    let details = load_geoffroy_hinton_transformer(&device).unwrap();
    test_model(details);
}
