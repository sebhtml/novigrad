use crate::{
    tensor::Error, Device, Embedding, Linear, Model, Reshape, Sigmoid, Softmax, TensorWithGrad,
    UnaryModel, UnaryOperator, WeightsInitialization,
};

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

impl UnaryModel for SimpleModel {}

impl SimpleModel {
    pub fn new(device: &Device, sequence_length: usize, vocab_size: usize) -> Result<Self, Error> {
        let n_embd = 384;
        let output_rows = 1;

        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let linear_0 = Linear::new(
            device,
            n_embd,
            n_embd,
            WeightsInitialization::Kaiming,
            sequence_length,
        )?;
        let sigmoid_0 = Sigmoid::new(device);
        let reshape = Reshape::new(
            device,
            vec![sequence_length, n_embd],
            vec![output_rows, sequence_length * n_embd],
        );
        let linear_1 = Linear::new(
            device,
            n_embd,
            sequence_length * n_embd,
            WeightsInitialization::Kaiming,
            output_rows,
        )?;
        let sigmoid_1 = Sigmoid::new(device);
        let linear_2 = Linear::new(
            device,
            vocab_size,
            n_embd,
            WeightsInitialization::Kaiming,
            output_rows,
        )?;
        let softmax = Softmax::new_with_next_is_cross_entropy_loss(device);

        let model = Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![output_rows, vocab_size],
            embedding,
            linear_0,
            sigmoid_0,
            reshape,
            linear_1,
            sigmoid_1,
            linear_2,
            softmax,
        };
        Ok(model)
    }
}

impl UnaryOperator for SimpleModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
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
        self.input_shape.clone()
    }
    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}
