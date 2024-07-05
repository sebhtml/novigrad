use crate::{
    display::RawPrinter, new_tensor_with_grad, perceptron::PerceptronModel,
    stochastic_gradient_descent::StochasticGradientDescent,
    sum_of_squared_errors::SumOfSquaredErrors, tensor::Error, Device, Metrics, TensorWithGrad,
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

pub fn load_addition_perceptron(
    device: &Device,
) -> Result<
    DatasetDetails<PerceptronModel, SumOfSquaredErrors, StochasticGradientDescent, RawPrinter>,
    Error,
> {
    let model = PerceptronModel::new(device)?;
    let examples = load_examples(device)?;
    let loss_operator = SumOfSquaredErrors::new(device);
    let optimizer = StochasticGradientDescent::new(0.5);
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 100,
        progress: 10,
        shuffle_examples: false,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics { total_loss: 0.1 },
        final_metrics_max: Metrics { total_loss: 15.0 },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: RawPrinter::default(),
        batch_size: 1,
    };
    Ok(details)
}
