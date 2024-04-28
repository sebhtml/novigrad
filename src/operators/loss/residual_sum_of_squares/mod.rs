use std::ops::Deref;

use crate::{devices::Device, DeltaWorkingMemory, Error, LearningTensor, OperatorTrait, TensorF32};

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
        let mut diffs = device.tensor(0, 0, vec![]);
        diffs.assign(device, expected)?;
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
        result.assign(device, expected)?;
        TensorF32::sub(device, actual, result)?;
        TensorF32::scalar_mul(device, -2.0, result)
    }
}

impl OperatorTrait for ResidualSumOfSquares {
    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &[LearningTensor],
        _output: &LearningTensor,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let actual: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let backward_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
        self.derive(device, expected, actual, backward_gradient)?;
        Ok(())
    }

    fn forward(&self, device: &Device, inputs: &[LearningTensor]) -> Result<LearningTensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output = device.learning_tensor(0, 0, vec![], false);
        let expected: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let actual: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let loss = self.evaluate(device, expected, actual)?;
        output.tensor().deref().borrow_mut().reset(1, 1, loss)?;
        Ok(output)
    }

    fn name(&self) -> &str {
        "ResidualSumOfSquares"
    }
}
