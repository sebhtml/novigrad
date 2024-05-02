use std::ops::Deref;

use crate::{devices::Device, Error, OperatorTrait, Tensor, TensorF32};

use super::LossFunction;

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct ResidualSumOfSquares {}

impl Default for ResidualSumOfSquares {
    fn default() -> Self {
        Self {}
    }
}

impl LossFunction for ResidualSumOfSquares {
    /// RSS = Σ (y_i - f(x_i))^2
    fn evaluate(
        &self,
        device: &Device,
        expected: &TensorF32,
        actual: &TensorF32,
    ) -> Result<f32, Error> {
        if expected.shape() != actual.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let rows = expected.rows();
        let cols = expected.cols();
        let len = rows * cols;
        let mut diffs = device.tensor_f32(rows, cols, vec![0.0; len]);
        TensorF32::copy(device, expected, &mut diffs)?;
        TensorF32::sub(device, actual, &mut diffs)?;
        TensorF32::dot_product(device, &diffs, &diffs)
    }

    fn derive(
        &self,
        device: &Device,
        expected: &TensorF32,
        actual: &TensorF32,
        result: &mut TensorF32,
    ) -> Result<(), Error> {
        TensorF32::copy(device, expected, result)?;
        TensorF32::sub(device, actual, result)?;
        TensorF32::scalar_mul(device, -2.0, result)
    }
}

impl OperatorTrait for ResidualSumOfSquares {
    fn backward(&self, device: &Device, inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let actual: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let backward_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
        self.derive(device, expected, actual, backward_gradient)?;
        Ok(())
    }

    fn forward(&self, device: &Device, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let actual: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let loss = self.evaluate(device, expected, actual)?;
        let output = device.tensor(1, 1, vec![loss], false);
        Ok(output)
    }

    fn name(&self) -> &str {
        "ResidualSumOfSquares"
    }
}
