use crate::{
    gelu::Gelu, statistics::layer_norm::LayerNormalization, tensor::Error, Add, BinaryOperator,
    Device, Dropout, Linear, MultiHeadAttention, TensorWithGrad, TernaryOperator, UnaryOperator,
    WeightsInitialization,
};

/// See:
/// Attention Is All You Need
/// https://arxiv.org/abs/1706.03762
///
/// See:
/// Full GPT Architecture
/// https://en.wikipedia.org/wiki/GPT-1#/media/File:Full_GPT_architecture.svg
pub struct Transformer {
    layer_norm_1: LayerNormalization,
    multi_head_attention: MultiHeadAttention,
    dropout_1: Dropout,
    layer_norm_2: LayerNormalization,
    add: Add,
    linear_1: Linear,
    gelu: Gelu,
    linear_2: Linear,
    dropout_2: Dropout,
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
        let layer_norm_1 = LayerNormalization::try_new(device, rows, cols)?;
        let multi_head_attention = MultiHeadAttention::try_new(
            device,
            rows,
            cols,
            causal_mask,
            num_heads,
            dropout_probability,
        )?;
        let dropout_1 = Dropout::try_new(device, rows, cols, dropout_probability)?;
        let add = Add::new(device);
        let layer_norm_2 = LayerNormalization::try_new(device, rows, cols)?;

        let linear_1 = Linear::new(device, cols, cols, WeightsInitialization::Kaiming, rows)?;
        let gelu = Gelu::new(device);
        let linear_2 = Linear::new(device, cols, cols, WeightsInitialization::Kaiming, rows)?;
        let dropout_2 = Dropout::try_new(device, rows, cols, dropout_probability)?;

        let transformer = Self {
            layer_norm_1,
            multi_head_attention,
            dropout_1,
            layer_norm_2,
            add,
            linear_1,
            gelu,
            linear_2,
            dropout_2,
        };
        Ok(transformer)
    }
}

impl UnaryOperator for Transformer {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let normalized_input = self.layer_norm_1.forward(&input)?;
        let attended = self.multi_head_attention.forward(
            &normalized_input,
            &normalized_input,
            &normalized_input,
        )?;
        let with_dropout_1 = self.dropout_1.forward(&attended)?;
        let residual_1 = self.add.forward(&with_dropout_1, &input)?;
        let normalized_output = self.layer_norm_2.forward(&residual_1)?;
        let lin_1 = self.linear_1.forward(&normalized_output)?;
        let activated = self.gelu.forward(&lin_1)?;
        let lin_2 = self.linear_2.forward(&activated)?;
        let with_dropout_2 = self.dropout_2.forward(&lin_2)?;
        let residual_2 = self.add.forward(&with_dropout_2, &normalized_output)?;
        Ok(residual_2)
    }
}
