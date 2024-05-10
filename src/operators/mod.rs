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
use core::fmt::Debug;

pub trait Operator {
    fn name(&self) -> &str;
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error>;
    fn forward_realize(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error>;
    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error>;
}

impl Debug for dyn Operator {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}
