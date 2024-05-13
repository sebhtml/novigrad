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
    pub fn try_new(device: &Device, rows: usize, cols: usize) -> Result<Self, Error> {
        let q = Linear::new(device, cols, cols, rows);
        let k = Linear::new(device, cols, cols, rows);
        let v = Linear::new(device, cols, cols, rows);

        let mask = true;
        let attention = ScaledDotProductAttention::try_new(device, rows, cols, mask).unwrap();

        let head = Self { q, k, v, attention };
        Ok(head)
    }
}

impl UnaryOperator for AttentionHead {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let q = self.q.forward(input)?;
        let k = self.k.forward(input)?;
        let v = self.v.forward(input)?;
        let attentions = self.attention.forward(&q, &k, &v)?;
        Ok(attentions)
    }
}
