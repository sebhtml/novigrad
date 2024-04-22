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
    fn evaluate(
        &self,
        accelerator: &Device,
        expected: &Tensor,
        actual: &Tensor,
    ) -> Result<f32, Error> {
        if expected.shape() != actual.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let mut diffs = Tensor::new(0, 0, vec![0.0]);
        diffs.assign(accelerator, expected);
        Tensor::sub(accelerator, actual, &mut diffs)?;
        Tensor::dot_product(accelerator, &diffs, &diffs)
    }

    fn derive(
        &self,
        accelerator: &Device,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        result.assign(accelerator, expected);
        Tensor::sub(accelerator, actual, result)?;
        Tensor::scalar_mul(accelerator, -2.0, result);
        Ok(())
    }
}

impl OperatorTrait for ResidualSumOfSquares {
    fn backward(
        &self,
        accelerator: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<Tensor>>,
        _output: &Rc<Tensor>,
        back_propagated_delta: &mut Tensor,
        _layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[1];
        self.derive(accelerator, expected, actual, back_propagated_delta)?;

        Ok((back_propagated_delta.clone(), vec![]))
    }

    fn forward(&self, accelerator: &Device, inputs: &Vec<Rc<Tensor>>) -> Result<Rc<Tensor>, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[1];
        let loss = self.evaluate(accelerator, expected, actual)?;
        let output = Tensor::new(1, 1, vec![loss]);
        Ok(output.into())
    }

    fn name(&self) -> &str {
        "ResidualSumOfSquares"
    }
}
