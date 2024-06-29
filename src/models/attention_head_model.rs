use crate::{
    tensor::Error, AttentionHead, Device, Embedding, Linear, Model, Softmax, TensorWithGrad,
    TernaryOperator, UnaryModel, UnaryOperator, WeightsInitialization,
};

pub struct AttentionHeadModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    attention_head: AttentionHead,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for AttentionHeadModel {}

impl AttentionHeadModel {
    pub fn new(
        device: &Device,
        sequence_length: usize,
        vocab_size: usize,
        n_embd: usize,
        causal_mask: bool,
        dropout_probability: f32,
    ) -> Result<Self, Error> {
        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let attention_head = AttentionHead::try_new(
            device,
            sequence_length,
            n_embd,
            n_embd,
            causal_mask,
            dropout_probability,
        )?;
        let linear = Linear::new(
            device,
            vocab_size,
            n_embd,
            WeightsInitialization::Kaiming,
            sequence_length,
        )?;
        let softmax = Softmax::new_with_next_is_cross_entropy_loss(device);
        let model = Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            embedding,
            attention_head,
            linear,
            softmax,
        };
        Ok(model)
    }
}

impl UnaryOperator for AttentionHeadModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let embedding = self.embedding.forward(input)?;
        let attentions = self
            .attention_head
            .forward(&embedding, &embedding, &embedding)?;
        let linear = self.linear.forward(&attentions)?;
        let softmax = self.softmax.forward(&linear)?;
        Ok(softmax)
    }
}

impl Model for AttentionHeadModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}
