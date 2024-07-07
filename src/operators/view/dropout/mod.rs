use crate::{
    instruction, new_tensor, new_tensor_with_grad,
    opcode::OpCode,
    tensor::{Error, Tensor},
    Category, Device, OperatorAttributes, TensorWithGrad, UnaryOperator,
};

#[cfg(test)]
mod tests;

/// See:
/// Dropout: A Simple Way to Prevent Neural Networks from Overfitting
/// https://www.jmlr.org/papers/v15/srivastava14a.html
pub struct Dropout {
    device: Device,
    probability: f32,
    mask: Tensor,
    alpha: Tensor,
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
        let mask = new_tensor!(device, mask_rows, mask_cols, mask)?;
        let probability = 1.0 - dropout_probability;
        let alpha = 1.0 / (1.0 - dropout_probability);
        let alpha = new_tensor!(device, mask_rows, mask_cols, vec![alpha])?;
        let mask = Self {
            device: device.clone(),
            probability,
            mask,
            alpha,
        };
        Ok(mask)
    }
}

impl UnaryOperator for Dropout {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let rows = input.tensor().rows();
        let cols = input.tensor().cols();
        let len = rows * cols;
        let output = new_tensor_with_grad!(
            self.device,
            rows,
            cols,
            vec![0.0; len],
            &[input],
            true,
            false,
        )?;

        output.push_instruction(instruction!(
            OpCode::Bernoulli,
            OperatorAttributes::F32(self.probability),
            &[&self.mask],
            &[&self.mask],
            Category::EnableDropout,
        ));

        output.push_instruction(instruction!(
            OpCode::Bernoulli,
            OperatorAttributes::F32(1.0),
            &[&self.mask],
            &[&self.mask],
            Category::DisableDropout,
        ));

        output.push_instruction(instruction!(
            OpCode::Mul,
            OperatorAttributes::None,
            &[&self.mask, &input.tensor()],
            &[&output.tensor()],
            Category::Inference,
        ));
        output.push_instruction(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&self.alpha, &output.tensor()],
            &[&output.tensor()],
            Category::Inference,
        ));
        output.push_instruction(instruction!(
            OpCode::Mul,
            OperatorAttributes::None,
            &[&self.mask, &output.gradient()],
            &[&input.gradient()],
            Category::Gradient,
        ));
        Ok(output)
    }
}
