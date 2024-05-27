use std::ops::Deref;

use crate::{
    inference_instruction, BinaryOperator, Device, Error, Mul, OpCode, ScalarMul, TensorWithGrad,
    UnaryOperator,
};

#[cfg(test)]
mod tests;

pub struct Dropout {
    mask: TensorWithGrad,
    mul: Mul,
    scalar_mul: ScalarMul,
    dropout_probability: f32,
}

impl Dropout {
    pub fn try_new(
        device: &Device,
        mask_rows: usize,
        mask_cols: usize,
        dropout_probability: f32,
    ) -> Result<Self, Error> {
        let len = mask_rows * mask_cols;
        let mask = vec![1.0; len];
        let mask = device.tensor_with_grad(mask_rows, mask_cols, mask, &[], false, false)?;
        let mul = Mul::new(device);
        let alpha = 1.0 / (1.0 - dropout_probability);
        let scalar_mul = ScalarMul::new(device, alpha);
        let mask = Self {
            mask,
            mul,
            scalar_mul,
            dropout_probability,
        };
        Ok(mask)
    }
}

impl UnaryOperator for Dropout {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let mask = &self.mask;
        mask.push_instruction(inference_instruction!(
            OpCode::Bernoulli(1.0 - self.dropout_probability),
            &[&mask.tensor().deref().borrow()],
            &[&mask.tensor().deref().borrow()],
        ));
        let mul_output = self.mul.forward(input, &self.mask)?;
        let scalar_mul_output = self.scalar_mul.forward(&mul_output)?;
        Ok(scalar_mul_output)
    }
}
