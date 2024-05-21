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

use crate::{Error, Tensor};

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
