mod activation;
pub use activation::*;
mod layer;
pub use layer::*;
mod loss;
pub use loss::*;

use crate::{Error, Tensor};
use core::fmt::Debug;

pub trait OperatorTrait {
    fn name(&self) -> &str;
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error>;
    fn forward_realize(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error>;
    fn backward(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error>;
}

impl Debug for dyn OperatorTrait {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}
