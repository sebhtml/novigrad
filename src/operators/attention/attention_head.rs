use crate::{
    tensor::Error, Device, Linear, ScaledDotProductAttention, TensorWithGrad, TernaryOperator,
    UnaryOperator, WeightsInitialization,
};

pub struct AttentionHead {
    q: Linear,
    k: Linear,
    v: Linear,
    attention: ScaledDotProductAttention,
}

impl AttentionHead {
    pub fn try_new(
        device: &Device,
        rows: usize,
        cols: usize,
        head_cols: usize,
        mask: bool,
        dropout_probability: f32,
    ) -> Result<Self, Error> {
        let q = Linear::new(
            device,
            head_cols,
            cols,
            WeightsInitialization::Kaiming,
            rows,
        )?;
        let k = Linear::new(
            device,
            head_cols,
            cols,
            WeightsInitialization::Kaiming,
            rows,
        )?;
        let v = Linear::new(
            device,
            head_cols,
            cols,
            WeightsInitialization::Kaiming,
            rows,
        )?;
        let attention =
            ScaledDotProductAttention::try_new(device, rows, cols, mask, dropout_probability)
                .unwrap();

        let head = Self { q, k, v, attention };
        Ok(head)
    }
}

impl TernaryOperator for AttentionHead {
    fn forward(
        &self,
        q: &TensorWithGrad,
        k: &TensorWithGrad,
        v: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let q = self.q.forward(q)?;
        let k = self.k.forward(k)?;
        let v = self.v.forward(v)?;
        let attentions = self.attention.forward(&q, &k, &v)?;
        Ok(attentions)
    }
}
