use std::rc::Rc;

use crate::{devices::Device, DeltaWorkingMemory, Error, Gradient, OperatorTrait, Tensor};

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
        diffs.assign(device, expected);
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
        result.assign(device, expected);
        Tensor::sub(device, actual, result)?;
        Tensor::scalar_mul(device, -2.0, result);
        Ok(())
    }
}

impl OperatorTrait for ResidualSumOfSquares {
    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<Tensor>>,
        _output: &Rc<Tensor>,
        back_propagated_delta: &mut Tensor,
        _layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[1];
        self.derive(device, expected, actual, back_propagated_delta)?;

        Ok((back_propagated_delta.clone(), vec![]))
    }

    fn forward(&self, device: &Device, inputs: &Vec<Rc<Tensor>>) -> Result<Rc<Tensor>, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[1];
        let loss = self.evaluate(device, expected, actual)?;
        let output = device.tensor(1, 1, vec![loss]);
        Ok(output.into())
    }

    fn name(&self) -> &str {
        "ResidualSumOfSquares"
    }
}
