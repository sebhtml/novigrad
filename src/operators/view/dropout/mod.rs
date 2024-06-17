use crate::{
    inference_instruction, new_tensor, new_tensor_with_grad,
    tensor::{Error, Tensor},
    BinaryOperator, Device, Mul, OpCode, OperatorAttributes, ScalarMul, TensorWithGrad,
    UnaryOperator,
};

#[cfg(test)]
mod tests;

pub struct Dropout {
    probabilities: Tensor,
    mask: TensorWithGrad,
    mul: Mul,
    scalar_mul: ScalarMul,
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
        let mask = new_tensor_with_grad!(device, mask_rows, mask_cols, mask, &[], false, false)?;
        let probability = 1.0 - dropout_probability;
        let probabilities = vec![probability; len];
        let probabilities = new_tensor!(device, mask_rows, mask_cols, probabilities)?;
        let mul = Mul::new(device);
        let alpha = 1.0 / (1.0 - dropout_probability);
        let scalar_mul = ScalarMul::new(device, alpha);
        let mask = Self {
            probabilities,
            mask,
            mul,
            scalar_mul,
        };
        Ok(mask)
    }
}

impl UnaryOperator for Dropout {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let probabilities = &self.probabilities;
        let mask = &self.mask;
        mask.push_instruction(inference_instruction!(
            OpCode::Bernoulli,
            OperatorAttributes::None,
            &[probabilities],
            &[&mask.tensor()],
        ));
        let mul_output = self.mul.forward(input, &self.mask)?;
        let scalar_mul_output = self.scalar_mul.forward(&mul_output)?;
        Ok(scalar_mul_output)
    }
}
