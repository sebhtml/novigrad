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
    fn compute_gradients(
        &self,
        accelerator: &Accelerator,
        inputs: &Vec<Rc<Tensor>>,
        layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error>;

    fn forward(
        &self,
        accelerator: &Accelerator,
        inputs: &Vec<Rc<Tensor>>,
    ) -> Result<Rc<Tensor>, Error>;

    // TODO backward should return Error
    fn backward(
        &self,
        inputs: &Vec<Rc<Tensor>>,
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    );

    // TODO get_layer_delta should return Error
    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<Tensor>>,
        output: &Rc<Tensor>,
        back_propagated_layer_output_delta: &Tensor,
        layer_output_delta: &mut Tensor,
    );
}

pub trait Forward {
    fn forward(&mut self, input: &Rc<Tensor>) -> Result<Rc<Tensor>, Error>;
    fn accelerator(&self) -> Rc<Accelerator>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}
