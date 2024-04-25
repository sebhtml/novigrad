use std::{cell::RefCell, rc::Rc};

use crate::{devices::Device, DeltaWorkingMemory, Error};

mod tape;
pub use tape::*;
mod learning_tensor;
pub use learning_tensor::*;
mod back_propagation;
pub use back_propagation::*;

pub trait OperatorTrait {
    fn name(&self) -> &str;

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<LearningTensor>,
    ) -> Result<LearningTensor, Error>;

    fn backward(
        &self,
        device: &Device,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<LearningTensor>,
        output: &LearningTensor,
        enabled_gradients: &mut Vec<LearningTensor>,
    ) -> Result<(), Error>;
}

pub trait Forward {
    fn forward(&mut self, input: &LearningTensor) -> Result<LearningTensor, Error>;
    fn device(&self) -> Rc<Device>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}
