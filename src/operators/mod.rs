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

use crate::{Error, TensorWithGrad};

pub trait UnaryOperator {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error>;
}

pub trait BinaryOperator {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error>;
}

pub trait TernaryOperator {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
        input_3: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error>;
}

/// An n-ary function takes n arguments.
/// https://en.wikipedia.org/wiki/Arity
pub trait NaryOperator {
    fn forward(&self, inputs: &[&TensorWithGrad]) -> Result<TensorWithGrad, Error>;
}
