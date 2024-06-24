use crate::{
    display::RawPrinter, new_tensor_with_grad, perceptron::PerceptronModel, tensor::Error, Device,
    GradientDescent, Metrics, ReduceSumSquare, TensorWithGrad,
};

use super::DatasetDetails;

fn load_examples(device: &Device) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let examples = vec![
        (vec![2.0, 3.0], vec![5.0]),
        (vec![2.0, 2.0], vec![4.0]),
        (vec![2.0, 1.0], vec![3.0]),
    ];
    let examples = examples
        .into_iter()
        .filter_map(|(x, y)| {
            let x = new_tensor_with_grad!(device, 1, x.len(), x, &[], false, false).ok();
            let y = new_tensor_with_grad!(device, 1, y.len(), y, &[], false, false).ok();
            match (x, y) {
                (Some(x), Some(y)) => Some((x, y)),
                _ => None,
            }
        })
        .collect();
    Ok(examples)
}

pub fn load_addition_dataset(
    device: &Device,
) -> Result<DatasetDetails<PerceptronModel, ReduceSumSquare, GradientDescent, RawPrinter>, Error> {
    let model = PerceptronModel::new(device)?;
    let examples = load_examples(device)?;
    let loss_operator = ReduceSumSquare::new(device);
    let learning_rate = 0.5;
    let optimizer = GradientDescent::new(learning_rate);
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 100,
        progress: 10,
        learning_rate,
        shuffle_examples: false,
        clipped_gradient_norm: true,
        initial_metrics: Metrics {
            total_loss: 0.1,
            total_next_token_perplexity: f32::NAN,
        },
        final_metrics: Metrics {
            total_loss: 15.0,
            total_next_token_perplexity: 1.0,
        },
        maximum_incorrect_argmaxes: 0,
        printer: RawPrinter::default(),
    };
    Ok(details)
}
