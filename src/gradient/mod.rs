use std::{cell::RefCell, rc::Rc};

use crate::{devices::Device, DeltaWorkingMemory, Error};

mod tape;
pub use tape::*;
mod learning_tensor;
pub use learning_tensor::*;

pub trait OperatorTrait {
    fn name(&self) -> &str;

    fn forward(&self, device: &Device, inputs: &[Tensor]) -> Result<Tensor, Error>;

    fn backward(
        &self,
        device: &Device,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &[Tensor],
        output: &Tensor,
    ) -> Result<(), Error>;
}

pub trait Forward {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error>;

    fn device(&self) -> Rc<Device>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}
