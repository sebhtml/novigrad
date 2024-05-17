use super::load_examples;
use crate::{BinaryOperator, CrossEntropyLoss, Device, Tokenizer, UnaryModel, UnaryOperator};
use crate::{Error, ModelDetails};

use crate::{Embedding, Linear, MatMul, Model, Reshape, Softmax, Tensor};

struct MegaManModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    vocab_size: usize,
    sequence_length: usize,
    parameters: Tensor,
    embedding: Embedding,
    matmul: MatMul,
    reshape: Reshape,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for MegaManModel {}

impl MegaManModel {
    pub fn new(device: &Device) -> Self {
        let sequence_length = 32;
        let vocab_size = 256;
        let n_embd = 384;
        let output_rows = 1;

        Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![output_rows, vocab_size],
            vocab_size,
            sequence_length,
            parameters: device.tensor(n_embd, n_embd, vec![0.0; n_embd * n_embd], &[], true, true),
            embedding: Embedding::new(device, vocab_size, n_embd),
            matmul: MatMul::new(device, true),
            reshape: Reshape::new(
                device,
                vec![sequence_length, n_embd],
                vec![output_rows, sequence_length * n_embd],
            ),
            linear: Linear::new(
                device,
                vocab_size,
                sequence_length * n_embd,
                true,
                output_rows,
            ),
            softmax: Softmax::new(device),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

impl UnaryOperator for MegaManModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let state_0 = self.embedding.forward(input)?;
        let state_0b = self.matmul.forward(&state_0, &self.parameters)?;
        let state_1 = self.reshape.forward(&state_0b)?;
        let state_2 = self.linear.forward(&state_1)?;
        let state_3 = self.softmax.forward(&state_2)?;
        Ok(state_3)
    }
}

impl Model for MegaManModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}

pub fn load_mega_man_model(device: &Device) -> Result<ModelDetails, Error> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let max_number_of_examples = 10;
    let model = MegaManModel::new(device);
    let vocab_size = model.vocab_size();
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let input_sequence_length = model.sequence_length();
    let output_sequence_length = 1;
    let examples = load_examples(
        &device,
        file_path,
        max_chars,
        max_number_of_examples,
        input_sequence_length,
        output_sequence_length,
        vocab_size,
        &mut tokenizer,
    )?;
    let loss_operator = CrossEntropyLoss::new(device);

    let details = ModelDetails {
        device: device.clone(),
        tokenizer: Some(tokenizer),
        examples,
        model: Box::new(model),
        loss_operator: Box::new(loss_operator),
        epochs: 100,
        progress: 10,
        initial_total_error_min: 50.0,
        final_total_error_max: 0.0,
        learning_rate: 0.5,
        clipped_gradient_norm: 1.0,
    };
    Ok(details)
}
