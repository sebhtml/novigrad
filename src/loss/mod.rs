use crate::{Error, Tensor};
mod residual_sum_of_squares;
pub use residual_sum_of_squares::*;
mod cross_entropy_loss;
pub use cross_entropy_loss::*;

pub trait LossFunction {
    fn evaluate(&self, expected: &Tensor, actual: &Tensor) -> Result<f32, Error>;
    fn derive(&self, expected: &Tensor, actual: &Tensor, result: &mut Tensor) -> Result<(), Error>;
}

#[derive(Clone, PartialEq)]
pub enum LossFunctionName {
    ResidualSumOfSquares,
    CrossEntropyLoss,
}

impl Into<Box<dyn LossFunction>> for &LossFunctionName {
    fn into(self) -> Box<dyn LossFunction> {
        match self {
            LossFunctionName::ResidualSumOfSquares => Box::new(ResidualSumOfSquares::default()),
            LossFunctionName::CrossEntropyLoss => Box::new(CrossEntropyLoss::default()),
        }
    }
}
