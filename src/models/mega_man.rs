use crate::tensor::Error;
use crate::{
    new_tensor_with_grad, BinaryOperator, Device, Embedding, Linear, MatMul, Model, Reshape,
    Softmax, TensorWithGrad, UnaryModel, UnaryOperator, WeightsInitialization,
};

pub struct MegaManModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    parameters: TensorWithGrad,
    embedding: Embedding,
    matmul: MatMul,
    reshape: Reshape,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for MegaManModel {}

impl MegaManModel {
    pub fn new(device: &Device, sequence_length: usize, vocab_size: usize) -> Result<Self, Error> {
        let n_embd = 384;
        let output_rows = 1;

        let model = Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![output_rows, vocab_size],
            parameters: new_tensor_with_grad!(
                device,
                n_embd,
                n_embd,
                vec![0.0; n_embd * n_embd],
                &[],
                true,
                true,
            )?,
            embedding: Embedding::new(device, vocab_size, n_embd)?,
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
                WeightsInitialization::Kaiming,
                output_rows,
            )?,
            softmax: Softmax::new_with_next_is_cross_entropy_loss(device),
        };
        Ok(model)
    }
}

impl UnaryOperator for MegaManModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
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
