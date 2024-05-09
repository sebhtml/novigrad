use crate::{
    Device, Embedding, Error, Identity, Linear, MatMul, Model, OperatorTrait, Reshape, Softmax,
    Tensor,
};
use std::rc::Rc;

pub struct MegaManModel {
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
            parameters: device.tensor(
                Rc::new(Identity::new(device)),
                &vec![],
                n_embd,
                n_embd,
                vec![0.0; n_embd * n_embd],
                true,
                true,
            ),
            embedding: Embedding::new(device, vocab_size, n_embd),
            matmul: MatMul::new(device, true),
            reshape: Reshape::new(
                device,
                vec![sequence_length, n_embd],
                vec![output_rows, sequence_length * n_embd],
            ),
            linear: Linear::new(device, vocab_size, sequence_length * n_embd, output_rows),
            softmax: Softmax::new(device, true),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

impl Model for MegaManModel {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        let state_0 = self.embedding.forward(inputs)?;
        let state_0b = self.matmul.forward(&[&state_0, &self.parameters])?;
        let state_1 = self.reshape.forward(&[&state_0b])?;
        let state_2 = self.linear.forward(&[&state_1])?;
        let state_3 = self.softmax.forward(&[&state_2])?;
        Ok(state_3)
    }

    fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}
