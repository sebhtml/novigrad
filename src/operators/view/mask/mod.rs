use std::ops::Deref;

use crate::{BinaryOperator, Device, Error, Mul, Tensor, UnaryOperator};

#[cfg(test)]
mod tests;

/// Attention Is All You Need
/// https://arxiv.org/abs/1706.03762
#[derive(Clone)]
pub struct Mask {
    mask: Tensor,
    mul: Mul,
}

impl Mask {
    pub fn try_new(device: &Device, mask_rows: usize, mask_cols: usize) -> Result<Self, Error> {
        let len = mask_rows * mask_cols;
        let mask = vec![1.0; len];

        let mask = device.tensor(mask_rows, mask_cols, mask, &[], true, true);
        let mut values = mask.tensor().deref().borrow().get_values()?;
        for row in 0..mask_rows {
            for col in 0..mask_cols {
                if row <= col {
                    let index = mask.tensor().deref().borrow().index(row, col);
                    values[index] = 0.0;
                }
            }
        }
        mask.tensor().deref().borrow_mut().set_values(values);

        let mul = Mul::new(device);
        let mask = Self { mask, mul };
        Ok(mask)
    }
}

impl UnaryOperator for Mask {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        self.mul.forward(input, &self.mask)
    }
}
