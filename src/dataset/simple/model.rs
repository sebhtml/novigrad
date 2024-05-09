use crate::{
    Device, Embedding, Error, ErrorEnum, Linear, OperatorTrait, Reshape, Sigmoid, Softmax, Tensor,
};

pub struct Model {
    embedding: Embedding,
    linear_0: Linear,
    sigmoid_0: Sigmoid,
    reshape: Reshape,
    linear_1: Linear,
    sigmoid_1: Sigmoid,
    linear_2: Linear,
    softmax: Softmax,
}

impl Model {
    pub fn new(device: &Device) -> Self {
        let _batch_size = 1;
        let sequence_length = 6;
        let vocab_size = 256;
        let num_embeddings = vocab_size;
        let embedding_dim = 384;
        let _num_heads = 0;
        let output_rows = 1;

        Self {
            embedding: Embedding::new(device, num_embeddings, embedding_dim),
            linear_0: Linear::new(device, embedding_dim, embedding_dim, sequence_length),
            sigmoid_0: Sigmoid::new(device),
            reshape: Reshape::new(
                device,
                vec![sequence_length, embedding_dim],
                vec![output_rows, sequence_length * embedding_dim],
            ),
            linear_1: Linear::new(
                device,
                embedding_dim,
                sequence_length * embedding_dim,
                output_rows,
            ),
            sigmoid_1: Sigmoid::new(device),
            linear_2: Linear::new(device, vocab_size, embedding_dim, output_rows),
            softmax: Softmax::new(device, true),
        }
    }
}

impl OperatorTrait for Model {
    fn name(&self) -> &str {
        "SimpleModel"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let state_0: Tensor = self.embedding.forward(inputs)?;
        let state_1 = self.linear_0.forward(&[state_0])?;
        let state_2 = self.sigmoid_0.forward(&[state_1])?;
        let state_3 = self.reshape.forward(&[state_2])?;
        let state_4 = self.linear_1.forward(&[state_3])?;
        let state_5 = self.sigmoid_1.forward(&[state_4])?;
        let state_6 = self.linear_2.forward(&[state_5])?;
        let state_7 = self.softmax.forward(&[state_6])?;
        Ok(state_7)
    }

    fn forward_realize(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::new(
            file!(),
            line!(),
            column!(),
            ErrorEnum::UnsupportedOperation,
        ))
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::new(
            file!(),
            line!(),
            column!(),
            ErrorEnum::UnsupportedOperation,
        ))
    }
}
