use crate::{
    new_tensor_with_grad,
    tensor::{Error, Tensor},
    BinaryOperator, Device, Mul, TensorWithGrad, UnaryOperator,
};

#[cfg(test)]
mod tests;

/// Attention Is All You Need
/// https://arxiv.org/abs/1706.03762
pub struct Mask {
    mask: TensorWithGrad,
    mul: Mul,
}

impl Mask {
    pub fn try_new(device: &Device, mask_rows: usize, mask_cols: usize) -> Result<Self, Error> {
        let len = mask_rows * mask_cols;

        let mut values = vec![1.0; len];
        for row in 0..mask_rows {
            for col in 0..mask_cols {
                // Mask positions that are in the future.
                if col > row {
                    let index = Tensor::get_index(&[mask_rows, mask_cols], row, col);
                    values[index] = 0.0;
                }
            }
        }
        let mask = new_tensor_with_grad!(device, mask_rows, mask_cols, values, &[], false, false)?;

        let mul = Mul::new(device);
        let mask = Self { mask, mul };
        Ok(mask)
    }
}

impl UnaryOperator for Mask {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        self.mul.forward(input, &self.mask)
    }
}
