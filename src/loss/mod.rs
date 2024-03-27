use crate::{Error, Tensor};
mod residual_sum_of_squares;
pub use residual_sum_of_squares::*;

pub trait LossFunction {
    fn evaluate(&self, expected: &Tensor, actual: &Tensor) -> Result<f32, Error>;
    fn derive(&self, expected: &Tensor, actual: &Tensor, result: &mut Tensor) -> Result<(), Error>;
}

pub enum LossFunctionName {
    ResidualSumOfSquares,
}

impl Into<Box<dyn LossFunction>> for &LossFunctionName {
    fn into(self) -> Box<dyn LossFunction> {
        match self {
            LossFunctionName::ResidualSumOfSquares => Box::new(ResidualSumOfSquares::default()),
        }
    }
}
