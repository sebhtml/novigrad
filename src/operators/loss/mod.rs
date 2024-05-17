use crate::{devices::Device, BinaryOperator, Error, Operator, TensorF32};
mod residual_sum_of_squares;
pub use residual_sum_of_squares::*;
mod cross_entropy_loss;
pub use cross_entropy_loss::*;

pub trait LossFunction {
    fn evaluate(device: &Device, expected: &TensorF32, actual: &TensorF32) -> Result<f32, Error>;
    fn derive(expected: &TensorF32, actual: &TensorF32, result: &TensorF32) -> Result<(), Error>;
}

pub trait LossOperator: BinaryOperator + Operator + LossFunction {}
