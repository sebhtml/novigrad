use crate::{Error, Tensor};
mod residual_sum_of_squares;
pub use residual_sum_of_squares::*;
mod cross_entropy_loss;
pub use cross_entropy_loss::*;

pub trait LossFunction {
    fn evaluate(&self, expected: &Tensor, actual: &Tensor) -> Result<f32, Error>;
    fn derive(
        &self,
        tmp: &mut Tensor,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error>;
}

pub enum LossFunctionType {
    ResidualSumOfSquares(ResidualSumOfSquares),
    CrossEntropyLoss(CrossEntropyLoss),
}

impl LossFunction for LossFunctionType {
    fn evaluate(&self, expected: &Tensor, actual: &Tensor) -> Result<f32, Error> {
        match self {
            LossFunctionType::ResidualSumOfSquares(that) => that.evaluate(expected, actual),
            LossFunctionType::CrossEntropyLoss(that) => that.evaluate(expected, actual),
        }
    }

    fn derive(
        &self,
        tmp: &mut Tensor,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            LossFunctionType::ResidualSumOfSquares(that) => {
                that.derive(tmp, expected, actual, result)
            }
            LossFunctionType::CrossEntropyLoss(that) => that.derive(tmp, expected, actual, result),
        }
    }
}
