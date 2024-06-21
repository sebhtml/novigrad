use crate::{
    tensor::Error, Device, Dropout, MultiHeadAttention, TensorWithGrad, TernaryOperator,
    UnaryOperator,
};

/// See:
/// Attention Is All You Need
/// https://arxiv.org/abs/1706.03762
pub struct Transformer {
    multi_head_attention: MultiHeadAttention,
    dropout: Dropout,
}

impl Transformer {
    pub fn try_new(
        device: &Device,
        rows: usize,
        cols: usize,
        causal_mask: bool,
        num_heads: usize,
        dropout_probability: f32,
    ) -> Result<Self, Error> {
        let multi_head_attention = MultiHeadAttention::try_new(
            device,
            rows,
            cols,
            causal_mask,
            num_heads,
            dropout_probability,
        )?;
        let dropout = Dropout::try_new(device, rows, cols, dropout_probability)?;

        let transformer = Self {
            multi_head_attention,
            dropout,
        };
        Ok(transformer)
    }
}

impl UnaryOperator for Transformer {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let x = self.multi_head_attention.forward(&input, &input, &input)?;
        let w = self.dropout.forward(&x)?;
        Ok(w)
    }
}
