use std::ops::Deref;

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    inference_instruction, BinaryOperator, Device, Error, Mul, OpCode, ScalarMul, Tensor,
    TensorWithGrad, UnaryOperator,
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
        let mask = device.tensor_with_grad(mask_rows, mask_cols, mask, &[], false, false);
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

    pub fn execute(
        dropout_probability: f32,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<(), Error> {
        // zero each element of the mask with probability p.
        let input = inputs[0];
        let output = outputs[0];
        let len = input.len();
        let mut values = vec![1.0; len];
        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0, 1.0);

        for i in 0..len {
            if rng.sample(uniform) < dropout_probability {
                values[i] = 0.0;
            }
        }
        output.set_values(values);
        Ok(())
    }
}

impl UnaryOperator for Dropout {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let mask = &self.mask;
        mask.push_instruction(inference_instruction!(
            OpCode::Dropout(self.dropout_probability),
            &[&mask.tensor().deref().borrow()],
            &[&mask.tensor().deref().borrow()],
        ));
        let mul_output = self.mul.forward(input, &self.mask)?;
        let scalar_mul_output = self.scalar_mul.forward(&mul_output)?;
        Ok(scalar_mul_output)
    }
}
