use std::{cell::RefCell, rc::Rc};

use crate::{devices::Device, DeltaWorkingMemory, Error, Tensor};

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
        device: &Device,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
    ) -> Result<Rc<RefCell<Tensor>>, Error>;

    fn backward(
        &self,
        device: &Device,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
        output: &Rc<RefCell<Tensor>>,
        back_propagated_delta: &Tensor,
        layer_delta: &mut Tensor,
    ) -> Result<(Rc<RefCell<Tensor>>, Vec<Gradient>), Error>;
}

pub trait Forward {
    fn forward(&mut self, input: &Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>, Error>;
    fn device(&self) -> Rc<Device>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}
