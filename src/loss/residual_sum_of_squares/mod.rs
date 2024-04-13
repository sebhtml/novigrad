use crate::{Error, Tensor};

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
    fn evaluate(&self, expected: &Tensor, actual: &Tensor) -> Result<f32, Error> {
        if expected.shape() != actual.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let mut diffs = Tensor::default();
        diffs.assign(expected);
        Tensor::saxpy(-1.0, actual, &mut diffs)?;
        Tensor::sdot(&diffs, &diffs)
    }

    fn derive(&self, expected: &Tensor, actual: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        result.assign(expected);
        Tensor::saxpy(-1.0, actual, result)?;
        Tensor::sscal(-2.0, result);
        Ok(())
    }
}
