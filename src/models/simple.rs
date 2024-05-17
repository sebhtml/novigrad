use crate::{
    into_one_hot_encoded_rows, CrossEntropyLoss, Device, Error, ErrorEnum, ModelDetails, Tensor,
    Tokenizer, TokenizerTrait, UnaryModel, UnaryOperator,
};

use crate::{Embedding, Linear, Model, Reshape, Sigmoid, Softmax};

struct SimpleModel {
    sequence_length: usize,
    vocab_size: usize,
    output_rows: usize,

    embedding: Embedding,
    linear_0: Linear,
    sigmoid_0: Sigmoid,
    reshape: Reshape,
    linear_1: Linear,
    sigmoid_1: Sigmoid,
    linear_2: Linear,
    softmax: Softmax,
}

impl UnaryModel for SimpleModel {}

impl SimpleModel {
    pub fn new(device: &Device) -> Self {
        let sequence_length = 6;
        let vocab_size = 256;
        let n_embd = 384;
        let output_rows = 1;

        let embedding = Embedding::new(device, vocab_size, n_embd);
        let linear_0 = Linear::new(device, n_embd, n_embd, true, sequence_length);
        let sigmoid_0 = Sigmoid::new(device);
        let reshape = Reshape::new(
            device,
            vec![sequence_length, n_embd],
            vec![output_rows, sequence_length * n_embd],
        );
        let linear_1 = Linear::new(device, n_embd, sequence_length * n_embd, true, output_rows);
        let sigmoid_1 = Sigmoid::new(device);
        let linear_2 = Linear::new(device, vocab_size, n_embd, true, output_rows);
        let softmax = Softmax::new(device);

        Self {
            sequence_length,
            vocab_size,
            output_rows,
            embedding,
            linear_0,
            sigmoid_0,
            reshape,
            linear_1,
            sigmoid_1,
            linear_2,
            softmax,
        }
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl UnaryOperator for SimpleModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let state_0 = self.embedding.forward(input)?;
        let state_1 = self.linear_0.forward(&state_0)?;
        let state_2 = self.sigmoid_0.forward(&state_1)?;
        let state_3 = self.reshape.forward(&state_2)?;
        let state_4 = self.linear_1.forward(&state_3)?;
        let state_5 = self.sigmoid_1.forward(&state_4)?;
        let state_6 = self.linear_2.forward(&state_5)?;
        let state_7 = self.softmax.forward(&state_6)?;
        Ok(state_7)
    }
}

impl Model for SimpleModel {
    fn input_size(&self) -> Vec<usize> {
        vec![self.sequence_length, self.vocab_size]
    }

    fn output_size(&self) -> Vec<usize> {
        vec![self.output_rows, self.vocab_size]
    }
}

fn load_examples(
    device: &Device,
    tokenizer: &mut Tokenizer,
    vocab_size: usize,
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

    let examples = examples
        .into_iter()
        .map(|example| {
            let one_hot_encoded_input = into_one_hot_encoded_rows(device, &example.0, vocab_size);
            let one_hot_encoded_output = into_one_hot_encoded_rows(device, &example.1, vocab_size);
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

pub fn load_simple_model(device: &Device) -> Result<ModelDetails, Error> {
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let model = SimpleModel::new(device);
    let examples = load_examples(&device, &mut tokenizer, model.vocab_size())?;
    let loss_operator = CrossEntropyLoss::new(device);
    let details = ModelDetails {
        device: device.clone(),
        tokenizer: Some(tokenizer),
        examples,
        model: Box::new(model),
        loss_operator: Box::new(loss_operator),
        epochs: 1000,
        progress: 100,
        initial_total_error_min: 8.0,
        final_total_error_max: 0.001,
        learning_rate: 0.5,
    };
    Ok(details)
}
