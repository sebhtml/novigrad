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
        blas: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
    ) -> Result<f32, Error> {
        if expected.shape() != actual.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let mut diffs = Tensor::default();
        diffs.assign(blas, expected);
        Tensor::saxpy(blas, -1.0, actual, &mut diffs)?;
        Tensor::sdot(blas, &diffs, &diffs)
    }

    fn derive(
        &self,
        blas: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        result.assign(blas, expected);
        Tensor::saxpy(blas, -1.0, actual, result)?;
        Tensor::sscal(blas, -2.0, result);
        Ok(())
    }
}
