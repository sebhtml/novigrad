use crate::{
    new_tensor_with_grad,
    tensor::{Error, Tensor},
    Add, BinaryOperator, Device, TensorWithGrad, UnaryOperator,
};

#[cfg(test)]
mod tests;

/// Attention Is All You Need
/// https://arxiv.org/abs/1706.03762
pub struct Mask {
    mask: TensorWithGrad,
    add: Add,
}

impl Mask {
    pub fn try_new(device: &Device, mask_rows: usize, mask_cols: usize) -> Result<Self, Error> {
        let len = mask_rows * mask_cols;

        let mut values = vec![0.0; len];
        for row in 0..mask_rows {
            for col in 0..mask_cols {
                // Mask positions that are in the future.
                if col > row {
                    let index = Tensor::get_index(&[mask_rows, mask_cols], row, col);
                    values[index] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = new_tensor_with_grad!(device, mask_rows, mask_cols, values, &[], true, true)?;

        let add = Add::new(device);
        let mask = Self { mask, add };
        Ok(mask)
    }
}

impl UnaryOperator for Mask {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        self.add.forward(input, &self.mask)
    }
}
