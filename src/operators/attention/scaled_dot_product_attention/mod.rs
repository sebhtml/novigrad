use crate::{
    BinaryOperator, Device, Dropout, Error, Mask, MatMul, ScalarMul, Softmax, Tensor,
    TernaryOperator, UnaryOperator,
};

#[cfg(test)]
mod tests;

/// Attention Is All You Need
/// https://arxiv.org/abs/1706.03762
pub struct ScaledDotProductAttention {
    qk_matmul: MatMul,
    scale: ScalarMul,
    mask: Option<Mask>,
    softmax: Softmax,
    dropout: Option<Dropout>,
    matmul: MatMul,
}

impl ScaledDotProductAttention {
    pub fn try_new(
        device: &Device,
        rows: usize,
        cols: usize,
        mask: bool,
        dropout_probability: f32,
    ) -> Result<Self, Error> {
        let qk_matmul = MatMul::new(device, true);
        let alpha = 1.0 / f32::sqrt(cols as f32);
        let scale = ScalarMul::new(device, alpha);
        let mask = match mask {
            false => None,
            true => {
                let mask = Mask::try_new(device, rows, rows)?;
                Some(mask)
            }
        };
        let softmax = Softmax::new(device);
        let dropout = if dropout_probability == 0.0 {
            None
        } else {
            Some(Dropout::try_new(device, rows, rows, dropout_probability)?)
        };
        let matmul = MatMul::new(device, false);

        let attention = Self {
            qk_matmul,
            scale,
            mask,
            softmax,
            dropout,
            matmul,
        };
        Ok(attention)
    }
}

impl TernaryOperator for ScaledDotProductAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
        let weights = self.qk_matmul.forward(q, k)?;
        let scaled_weights = self.scale.forward(&weights)?;
        let masked_weights = match &self.mask {
            Some(mask) => mask.forward(&scaled_weights)?,
            _ => scaled_weights,
        };
        let softmaxed_weights = self.softmax.forward(&masked_weights)?;
        let with_dropout = match &self.dropout {
            Some(dropout) => dropout.forward(&softmaxed_weights)?,
            _ => softmaxed_weights,
        };
        let attentions = self.matmul.forward(&with_dropout, v)?;
        Ok(attentions)
    }
}
