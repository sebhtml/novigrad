use crate::{Error, Tensor};

use super::LossFunction;

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
        let cols = expected.cols();
        let mut sum = 0.0;
        let last_row = actual.rows() - 1;
        if actual.cols() != cols {
            return Err(Error::IncompatibleTensorShapes);
        }
        for col in 0..cols {
            let diff = expected.get(0, col) - actual.get(last_row, col);
            sum += diff * diff;
        }
        Ok(sum)
    }

    fn derive(
        &self,
        tmp: &mut Tensor,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        expected.sub(actual, tmp)?;
        tmp.scalar_mul(-2.0, result)
    }
}
