mod activation;
pub use activation::*;
mod view;
pub use view::*;
mod loss;
pub use loss::*;
mod lin_alg;
pub use lin_alg::*;
mod attention;
pub use attention::*;

use crate::{Error, Tensor, TensorF32};
use core::fmt::Debug;

pub trait UnaryOperator {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error>;
}

pub trait BinaryOperator {
    fn forward(&self, input_1: &Tensor, input_2: &Tensor) -> Result<Tensor, Error>;
}

pub trait TernaryOperator {
    fn forward(
        &self,
        input_1: &Tensor,
        input_2: &Tensor,
        input_3: &Tensor,
    ) -> Result<Tensor, Error>;
}

/// An n-ary function takes n arguments.
/// https://en.wikipedia.org/wiki/Arity
pub trait NaryOperator {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error>;
}

pub trait Operator {
    fn name(&self) -> &str;
    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error>;
    fn backward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error>;
}

impl Debug for dyn Operator {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}
