use crate::{
    Device, Error, Linear, ScaledDotProductAttention, Tensor, TernaryOperator, UnaryOperator,
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
    ) -> Result<Self, Error> {
        let q = Linear::new(device, head_cols, cols, true, rows);
        let k = Linear::new(device, head_cols, cols, true, rows);
        let v = Linear::new(device, head_cols, cols, true, rows);
        let attention = ScaledDotProductAttention::try_new(device, rows, cols, mask).unwrap();

        let head = Self { q, k, v, attention };
        Ok(head)
    }
}

impl TernaryOperator for AttentionHead {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
        let q = self.q.forward(q)?;
        let k = self.k.forward(k)?;
        let v = self.v.forward(v)?;
        let attentions = self.attention.forward(&q, &k, &v)?;
        Ok(attentions)
    }
}
