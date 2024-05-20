use crate::{
    BinaryOperator, Device, Error, Mask, MatMul, ScalarMul, Softmax, Tensor, TernaryOperator,
    UnaryOperator,
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
    matmul: MatMul,
}

impl ScaledDotProductAttention {
    pub fn try_new(device: &Device, rows: usize, cols: usize, mask: bool) -> Result<Self, Error> {
        let qk_matmul = MatMul::new(device, true);
        let alpha = 1.0 / f32::sqrt(cols as f32);
        let scale = ScalarMul::new(device, alpha);
        let mask = match mask {
            false => None,
            true => {
                let mask_rows = rows;
                let mask_cols = rows;
                let mask = Mask::try_new(device, mask_rows, mask_cols)?;
                Some(mask)
            }
        };
        let softmax = Softmax::new(device, false);
        let matmul = MatMul::new(device, false);

        let attention = Self {
            qk_matmul,
            scale,
            mask,
            softmax,
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
        let attentions = self.matmul.forward(&softmaxed_weights, v)?;
        Ok(attentions)
    }
}
