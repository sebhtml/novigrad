use crate::{
    into_one_hot_encoded_rows, CrossEntropyLoss, DatasetDetails, Device, Error, ErrorEnum, Program,
    Tensor, Tokenizer, TokenizerTrait,
};

use crate::{Embedding, Linear, Model, OperatorTrait, Reshape, Sigmoid, Softmax};

pub struct SimpleModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    linear_0: Linear,
    sigmoid_0: Sigmoid,
    reshape: Reshape,
    linear_1: Linear,
    sigmoid_1: Sigmoid,
    linear_2: Linear,
    softmax: Softmax,
}

impl SimpleModel {
    pub fn new(device: &Device) -> Self {
        let sequence_length = 6;
        let vocab_size = 256;
        let n_embd = 384;
        let output_rows = 1;

        let linear_1 = Linear::new(device, n_embd, sequence_length * n_embd, output_rows);

        Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![output_rows, vocab_size],
            embedding: Embedding::new(device, vocab_size, n_embd),
            linear_0: Linear::new(device, n_embd, n_embd, sequence_length),
            sigmoid_0: Sigmoid::new(device),
            reshape: Reshape::new(
                device,
                vec![sequence_length, n_embd],
                vec![output_rows, sequence_length * n_embd],
            ),
            linear_1,
            sigmoid_1: Sigmoid::new(device),
            linear_2: Linear::new(device, vocab_size, n_embd, output_rows),
            softmax: Softmax::new(device, true),
        }
    }
}

impl Model for SimpleModel {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        let state_0: Tensor = self.embedding.forward(inputs)?;
        let state_1 = self.linear_0.forward(&[&state_0])?;
        let state_2 = self.sigmoid_0.forward(&[&state_1])?;
        let state_3 = self.reshape.forward(&[&state_2])?;
        let state_4 = self.linear_1.forward(&[&state_3])?;
        let state_5 = self.sigmoid_1.forward(&[&state_4])?;
        let state_6 = self.linear_2.forward(&[&state_5])?;
        let state_7 = self.softmax.forward(&[&state_6])?;
        Ok(state_7)
    }

    fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}

fn load_examples(
    device: &Device,
    tokenizer: &mut Tokenizer,
) -> Result<Vec<(Tensor, Tensor)>, Error> {
    let examples: Vec<_> = ["quizzed", "fuzzing"]
        .iter()
        .map(|text| {
            (
                tokenizer.encode(&text[0..text.len() - 1]),
                tokenizer.encode(&text[text.len() - 1..text.len()]),
            )
        })
        .collect();

    let num_classes = 256;
    let examples = examples
        .into_iter()
        .map(|example| {
            let one_hot_encoded_input = into_one_hot_encoded_rows(device, &example.0, num_classes);
            let one_hot_encoded_output = into_one_hot_encoded_rows(device, &example.1, num_classes);
            (one_hot_encoded_input, one_hot_encoded_output)
        })
        .try_fold(vec![], |mut acc, item| match item {
            (Ok(a), Ok(b)) => {
                acc.push((a, b));
                Ok(acc)
            }
            _ => Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::UnsupportedOperation,
            )),
        });

    examples
}

pub fn load_dataset(device: &Device) -> Result<DatasetDetails, Error> {
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let examples = load_examples(&device, &mut tokenizer)?;
    let model = SimpleModel::new(device);
    let loss_operator = CrossEntropyLoss::new(device);
    let program = Program::try_new(&device, &model, &loss_operator)?;
    let details = DatasetDetails {
        device: device.clone(),
        tokenizer,
        examples,
        program,
        epochs: 1000,
        progress: 100,
        initial_total_error_min: 8.0,
        final_total_error_max: 0.001,
        learning_rate: 0.5,
    };
    Ok(details)
}
