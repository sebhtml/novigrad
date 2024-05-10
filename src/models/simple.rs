use crate::{
    into_one_hot_encoded_rows, CrossEntropyLoss, DatasetDetails, Device, Error, ErrorEnum, Program,
    Tensor, Tokenizer, TokenizerTrait,
};

use crate::{Embedding, Linear, Model, Operator, Reshape, Sigmoid, Softmax};

struct SimpleModel {
    device: Device,
    sequence_length: usize,
    vocab_size: usize,
    n_embd: usize,
    output_rows: usize,
}

impl SimpleModel {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            sequence_length: 6,
            vocab_size: 256,
            n_embd: 384,
            output_rows: 1,
        }
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl Model for SimpleModel {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        let device = &self.device;
        let sequence_length = self.sequence_length;
        let vocab_size = self.vocab_size;
        let n_embd = self.n_embd;
        let output_rows = self.output_rows;

        let embedding = Embedding::new(device, vocab_size, n_embd);
        let linear_0 = Linear::new(device, n_embd, n_embd, sequence_length);
        let sigmoid_0 = Sigmoid::new(device);
        let reshape = Reshape::new(
            device,
            vec![sequence_length, n_embd],
            vec![output_rows, sequence_length * n_embd],
        );
        let linear_1 = Linear::new(device, n_embd, sequence_length * n_embd, output_rows);
        let sigmoid_1 = Sigmoid::new(device);
        let linear_2 = Linear::new(device, vocab_size, n_embd, output_rows);
        let softmax = Softmax::new(device, true);

        let state_0 = embedding.forward(inputs)?;
        let state_1 = linear_0.forward(&[&state_0])?;
        let state_2 = sigmoid_0.forward(&[&state_1])?;
        let state_3 = reshape.forward(&[&state_2])?;
        let state_4 = linear_1.forward(&[&state_3])?;
        let state_5 = sigmoid_1.forward(&[&state_4])?;
        let state_6 = linear_2.forward(&[&state_5])?;
        let state_7 = softmax.forward(&[&state_6])?;
        Ok(state_7)
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![self.sequence_length, self.vocab_size]
    }

    fn output_shape(&self) -> Vec<usize> {
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

pub fn load_dataset(device: &Device) -> Result<DatasetDetails, Error> {
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let model = SimpleModel::new(device);
    let examples = load_examples(&device, &mut tokenizer, model.vocab_size())?;
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
