use crate::{accelerator::Accelerator, Error, Gradient, OperatorTrait, Tensor};

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
        accelerator: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
    ) -> Result<f32, Error> {
        if expected.shape() != actual.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let mut diffs = Tensor::default();
        diffs.assign(accelerator, expected);
        Tensor::sub(accelerator, actual, &mut diffs)?;
        Tensor::dot_product(accelerator, &diffs, &diffs)
    }

    fn derive(
        &self,
        accelerator: &Accelerator,
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
    fn compute_gradients(
        &mut self,
        _accelerator: &Accelerator,
        _inputs: &Vec<Tensor>,
        _layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error> {
        Ok(vec![])
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[1];
        let loss = self.evaluate(accelerator, expected, actual)?;
        let tensor = Tensor::new(1, 1, vec![loss]);
        output.assign(accelerator, &tensor);
        Ok(())
    }

    fn backward(
        &self,
        inputs: &Vec<Tensor>,
        accelerator: &Accelerator,
        _layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    ) {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[1];
        let op_result = self.derive(accelerator, expected, actual, previous_layer_output_delta);
        op_result.expect("Ok");
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        _working_memory: &mut crate::DeltaWorkingMemory,
        _inputs: &Vec<Tensor>,
        _layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        layer_output_delta: &mut Tensor,
    ) {
        layer_output_delta.assign(accelerator, back_propagated_layer_output_delta)
    }
}
