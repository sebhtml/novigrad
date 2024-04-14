use crate::{accelerator::Accelerator, Error, Tensor};

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
        Tensor::sdot(accelerator, &diffs, &diffs)
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
