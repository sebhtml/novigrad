use crate::{
    error, AttentionHead, Concat, Device, Error, ErrorEnum, Linear, NaryOperator, Tensor,
    TernaryOperator, UnaryOperator,
};

pub struct MultiHeadAttention {
    attention_heads: Vec<AttentionHead>,
    concat: Concat,
    linear: Linear,
}

impl MultiHeadAttention {
    pub fn try_new(
        device: &Device,
        rows: usize,
        cols: usize,
        mask: bool,
        num_heads: usize,
        dropout_probability: f32,
    ) -> Result<Self, Error> {
        if cols % num_heads > 0 {
            return Err(error!(ErrorEnum::IncorrectOperatorConfiguration));
        }
        let head_cols = cols / num_heads;
        let mut attention_heads = vec![];
        for _ in 0..num_heads {
            attention_heads.push(AttentionHead::try_new(
                device,
                rows,
                cols,
                head_cols,
                mask,
                dropout_probability,
            )?);
        }

        let concat = Concat::new(device);
        let linear = Linear::new(device, cols, cols, true, rows);

        let multi_head_attention = Self {
            attention_heads,
            concat,
            linear,
        };
        Ok(multi_head_attention)
    }
}

impl TernaryOperator for MultiHeadAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
        let mut attention_head_attentions = vec![];
        for attention_head in self.attention_heads.iter() {
            let attentions = attention_head.forward(q, k, v)?;
            attention_head_attentions.push(attentions);
        }

        let attention_head_attentions: Vec<_> = attention_head_attentions.iter().collect();
        let concat = self.concat.forward(&attention_head_attentions)?;
        let linear = self.linear.forward(&concat)?;
        Ok(linear)
    }
}
