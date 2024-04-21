use std::{cell::RefCell, rc::Rc};

use crate::{accelerator::Accelerator, DeltaWorkingMemory, Error, Tensor};

mod tape;
pub use tape::*;
mod gradient;
pub use gradient::*;
mod back_propagation;
pub use back_propagation::*;

pub trait OperatorTrait {
    fn name(&self) -> &str;

    fn forward(
        &self,
        accelerator: &Accelerator,
        inputs: &Vec<Rc<Tensor>>,
    ) -> Result<Rc<Tensor>, Error>;

    fn backward(
        &self,
        accelerator: &Accelerator,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<Tensor>>,
        output: &Rc<Tensor>,
        back_propagated_delta: &mut Tensor,
        layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error>;
}

pub trait Forward {
    fn forward(&mut self, input: &Rc<Tensor>) -> Result<Rc<Tensor>, Error>;
    fn accelerator(&self) -> Rc<Accelerator>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}
