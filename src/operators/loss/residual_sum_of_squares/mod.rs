use std::ops::Deref;

use crate::{devices::Device, DeltaWorkingMemory, Error, LearningTensor, OperatorTrait, Tensor};

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
    fn evaluate(&self, device: &Device, expected: &Tensor, actual: &Tensor) -> Result<f32, Error> {
        if expected.shape() != actual.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let mut diffs = device.tensor(0, 0, vec![]);
        diffs.assign(device, expected)?;
        Tensor::sub(device, actual, &mut diffs)?;
        Tensor::dot_product(device, &diffs, &diffs)
    }

    fn derive(
        &self,
        device: &Device,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        result.assign(device, expected)?;
        Tensor::sub(device, actual, result)?;
        Tensor::scalar_mul(device, -2.0, result)
    }
}

impl OperatorTrait for ResidualSumOfSquares {
    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<LearningTensor>,
        _output: &LearningTensor,
        _enabled_gradients: &mut Vec<LearningTensor>,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected: &Tensor = &inputs[0].tensor().deref().borrow();
        let actual: &Tensor = &inputs[1].tensor().deref().borrow();
        {
            let backward_gradient: &mut Tensor = &mut inputs[1].gradient().deref().borrow_mut();
            self.derive(device, expected, actual, backward_gradient)?;
        }

        Ok(())
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<LearningTensor>,
    ) -> Result<LearningTensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output = device.learning_tensor(0, 0, vec![], false);
        let expected: &Tensor = &inputs[0].tensor().deref().borrow();
        let actual: &Tensor = &inputs[1].tensor().deref().borrow();
        let loss = self.evaluate(device, expected, actual)?;
        {
            let output: &mut Tensor = &mut output.tensor().deref().borrow_mut();
            output.reset(1, 1, loss);
        }
        Ok(output)
    }

    fn name(&self) -> &str {
        "ResidualSumOfSquares"
    }
}
