use std::ops::Deref;

use crate::{
    into_one_hot_encoded_rows, CrossEntropyLoss, DatasetDetails, Device, Error, ErrorEnum,
    NeuralMachine, ResidualSumOfSquares, Tensor, Tokenizer, TokenizerTrait, UnaryOperator,
};

use crate::{Embedding, Linear, Model, Reshape, Sigmoid, Softmax};

struct PerceptronModel {
    linear: Linear,
}

impl PerceptronModel {
    pub fn new(device: &Device) -> Self {
        let linear = Linear::new(device, 1, 2, 1);

        Self { linear }
    }
}

impl UnaryOperator for PerceptronModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        println!("linear forward");
        assert_eq!(
            &vec![1, 2],
            &input
                .tensor()
                .deref()
                .borrow()
                .size()
                .deref()
                .borrow()
                .clone(),
        );
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

fn load_examples(device: &Device) -> Result<Vec<(Tensor, Tensor)>, Error> {
    let examples = vec![
        (vec![2.0, 3.0], vec![5.0]),
        (vec![2.0, 2.0], vec![4.0]),
        (vec![2.0, 1.0], vec![3.0]),
    ];
    let examples = examples
        .into_iter()
        .map(|(x, y)| {
            (
                device.tensor(1, x.len(), x, &[], false, false),
                device.tensor(1, y.len(), y, &[], false, false),
            )
        })
        .collect();
    Ok(examples)
}

pub fn load_perceptron(device: &Device) -> Result<DatasetDetails, Error> {
    let model = PerceptronModel::new(device);
    let examples = load_examples(&device)?;
    let loss_operator = ResidualSumOfSquares::new(device);
    let program = NeuralMachine::try_new(&device, &model, &loss_operator)?;
    let details = DatasetDetails {
        device: device.clone(),
        tokenizer: None,
        examples,
        program,
        epochs: 100,
        progress: 10,
        initial_total_error_min: 8.0,
        final_total_error_max: 0.001,
        learning_rate: 0.5,
    };
    Ok(details)
}
