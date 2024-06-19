use crate::{new_tensor_with_grad, Linear, Model, WeightsInitialization};
use crate::{
    tensor::Error, Device, GradientDescent, Metrics, ModelDetails, ReduceSumSquare, TensorWithGrad,
    UnaryModel, UnaryOperator,
};

pub struct PerceptronModel {
    linear: Linear,
}

impl UnaryModel for PerceptronModel {}

impl PerceptronModel {
    pub fn new(device: &Device) -> Result<Self, Error> {
        let linear = Linear::new(device, 1, 2, WeightsInitialization::Kaiming, 1)?;
        let model = Self { linear };
        Ok(model)
    }
}

impl UnaryOperator for PerceptronModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        self.linear.forward(input)
    }
}

impl Model for PerceptronModel {
    fn input_size(&self) -> Vec<usize> {
        vec![1, 2]
    }
    fn output_size(&self) -> Vec<usize> {
        vec![1, 1]
    }
}

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

pub fn load_perceptron(
    device: &Device,
) -> Result<ModelDetails<PerceptronModel, ReduceSumSquare, GradientDescent>, Error> {
    let model = PerceptronModel::new(device)?;
    let examples = load_examples(device)?;
    let loss_operator = ReduceSumSquare::new(device);
    let learning_rate = 0.5;
    let optimizer = GradientDescent::new(learning_rate);
    let details = ModelDetails {
        device: device.clone(),
        tokenizer: None,
        examples,
        model,
        loss_operator,
        optimizer,
        epochs: 100,
        progress: 10,
        learning_rate,
        shuffle_examples: false,
        clipped_gradient_norm: 1.0,
        initial_metrics: Metrics {
            total_loss: 0.1,
            total_perplexity: f32::NAN,
        },
        final_metrics: Metrics {
            total_loss: 15.0,
            total_perplexity: 1.0,
        },
    };
    Ok(details)
}
