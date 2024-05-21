use crate::{BinaryOperator, Device, Error, Mul, Tensor, UnaryOperator};

#[cfg(test)]
mod tests;

pub struct Dropout {
    mask: Tensor,
    mul: Mul,
    _p: f32,
}

impl Dropout {
    pub fn try_new(
        device: &Device,
        mask_rows: usize,
        mask_cols: usize,
        p: f32,
    ) -> Result<Self, Error> {
        let len = mask_rows * mask_cols;
        let mask = vec![1.0; len];
        let mask = device.tensor(mask_rows, mask_cols, mask, &[], true, true);
        let mul = Mul::new(device);
        let mask = Self { mask, mul, _p: p };
        Ok(mask)
    }
}

impl UnaryOperator for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        // TODO zero each element of the mask with probability p.
        /*
        let mut values = mask.tensor().deref().borrow().get_values()?;
        let mut indices = (0..values.len()).collect()
        mask.tensor().deref().borrow_mut().set_values(values);
         */
        self.mul.forward(input, &self.mask)
        // TODO scale my 1 ( 1 - p )
    }
}
