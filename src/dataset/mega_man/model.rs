use crate::{
    Device, Embedding, Error, Identity, Linear, MatMul, OperatorTrait, Reshape, Softmax, Tensor,
};
use std::rc::Rc;

pub struct Model {
    vocab_size: usize,
    parameters: Tensor,
    embedding: Embedding,
    matmul: MatMul,
    reshape: Reshape,
    linear: Linear,
    softmax: Softmax,
}

impl Model {
    pub fn new(device: &Device) -> Self {
        let _batch_size = 1;
        let sequence_length = 32;
        let vocab_size = 256;
        //let vocab_size = 34816; // 32768 + 2048
        let num_embeddings = vocab_size;
        let embedding_dim = 384;
        let _num_heads = 0;
        let output_rows = 1;

        Self {
            vocab_size,
            parameters: device.tensor(
                Rc::new(Identity::new(device)),
                &vec![],
                embedding_dim,
                embedding_dim,
                vec![0.0; embedding_dim * embedding_dim],
                true,
            ),
            embedding: Embedding::new(device, num_embeddings, embedding_dim),
            matmul: MatMul::new(device),
            reshape: Reshape::new(
                device,
                vec![sequence_length, embedding_dim],
                vec![output_rows, sequence_length * embedding_dim],
            ),
            linear: Linear::new(
                device,
                vocab_size,
                sequence_length * embedding_dim,
                output_rows,
            ),
            softmax: Softmax::new(device, true),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl OperatorTrait for Model {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let state_0 = self.embedding.forward(inputs)?;
        let state_0b = self.matmul.forward(&[state_0, self.parameters.clone()])?;
        let state_1 = self.reshape.forward(&[state_0b])?;
        let state_2 = self.linear.forward(&[state_1])?;
        let state_3 = self.softmax.forward(&[state_2])?;
        Ok(state_3)
    }

    fn name(&self) -> &str {
        "MegaManModel"
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::UnsupportedOperation)
    }

    fn forward_realize(&self, _inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        output.realize()
    }
}
