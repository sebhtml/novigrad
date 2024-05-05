use crate::{devices::Device, Error, TensorF32};
mod residual_sum_of_squares;
pub use residual_sum_of_squares::*;
mod cross_entropy_loss;
pub use cross_entropy_loss::*;

pub trait LossFunction {
    fn evaluate(
        &self,
        device: &Device,
        expected: &TensorF32,
        actual: &TensorF32,
    ) -> Result<f32, Error>;
    fn derive(
        &self,
        expected: &TensorF32,
        actual: &TensorF32,
        result: &mut TensorF32,
    ) -> Result<(), Error>;
}
