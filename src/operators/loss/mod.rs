use crate::{accelerator::Accelerator, Error, Tensor};
mod residual_sum_of_squares;
pub use residual_sum_of_squares::*;
mod cross_entropy_loss;
pub use cross_entropy_loss::*;

pub trait LossFunction {
    fn evaluate(
        &self,
        accelerator: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
    ) -> Result<f32, Error>;
    fn derive(
        &self,
        accelerator: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error>;
}

#[derive(Clone)]
pub enum LossFunctionType {
    ResidualSumOfSquares(ResidualSumOfSquares),
    CrossEntropyLoss(CrossEntropyLoss),
}

impl LossFunction for LossFunctionType {
    fn evaluate(
        &self,
        accelerator: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
    ) -> Result<f32, Error> {
        match self {
            LossFunctionType::ResidualSumOfSquares(that) => {
                that.evaluate(accelerator, expected, actual)
            }
            LossFunctionType::CrossEntropyLoss(that) => {
                that.evaluate(accelerator, expected, actual)
            }
        }
    }

    fn derive(
        &self,
        accelerator: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            LossFunctionType::ResidualSumOfSquares(that) => {
                that.derive(accelerator, expected, actual, result)
            }
            LossFunctionType::CrossEntropyLoss(that) => {
                that.derive(accelerator, expected, actual, result)
            }
        }
    }
}
