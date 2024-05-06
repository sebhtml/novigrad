use crate::Error;
use core::fmt::Debug;

mod learning_tensor;
pub use learning_tensor::*;

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
