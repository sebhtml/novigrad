use crate::{devices::Device, Error};
use core::fmt::Debug;
use std::rc::Rc;

mod learning_tensor;
pub use learning_tensor::*;

pub trait OperatorTrait {
    fn name(&self) -> &str;

    fn forward(&self, device: &Device, inputs: &[Tensor]) -> Result<Tensor, Error>;

    fn backward(&self, device: &Device, inputs: &[Tensor], output: &Tensor) -> Result<(), Error>;
}

impl Debug for dyn OperatorTrait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub trait Forward {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error>;
    fn device(&self) -> Rc<Device>;
}
