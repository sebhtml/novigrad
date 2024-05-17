use std::ops::Deref;

use crate::{devices::Device, BinaryOperator, Error, Operator, TensorF32};
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
        result: &TensorF32,
    ) -> Result<(), Error>;
}

pub trait LossOperator: BinaryOperator + Operator + LossFunction {}

impl LossOperator for Box<dyn LossOperator> {}

impl BinaryOperator for Box<dyn LossOperator> {
    fn forward(
        &self,
        input_1: &crate::Tensor,
        input_2: &crate::Tensor,
    ) -> Result<crate::Tensor, Error> {
        BinaryOperator::forward(self.deref(), input_1, input_2)
    }
}

impl Operator for Box<dyn LossOperator> {
    fn name(&self) -> &str {
        self.deref().name()
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        Operator::forward(self.deref(), inputs, outputs)
    }
}

impl LossFunction for Box<dyn LossOperator> {
    fn evaluate(
        &self,
        device: &Device,
        expected: &TensorF32,
        actual: &TensorF32,
    ) -> Result<f32, Error> {
        self.deref().evaluate(device, expected, actual)
    }

    fn derive(
        &self,
        expected: &TensorF32,
        actual: &TensorF32,
        result: &TensorF32,
    ) -> Result<(), Error> {
        self.deref().derive(expected, actual, result)
    }
}
