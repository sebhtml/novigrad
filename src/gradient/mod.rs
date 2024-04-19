use std::{cell::RefCell, rc::Rc};

use crate::{accelerator::Accelerator, DeltaWorkingMemory, Error, Tensor};

mod tape;
pub use tape::*;
mod differentiable_tensor;
pub use differentiable_tensor::*;
mod back_propagation;
pub use back_propagation::*;

pub trait OperatorTrait {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    );

    fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32)
        -> Result<(), Error>;

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        output: &mut Tensor,
    ) -> Result<(), Error>;

    // TODO backward should return Error
    fn backward(
        &self,
        inputs: &Vec<Tensor>,
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    );

    // TODO get_layer_delta should return Error
    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Tensor>,
        layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        is_last_layer: bool,
        layer_output_delta: &mut Tensor,
    );
}

pub trait ForwardArchitecture {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error>;
    fn accelerator(&self) -> Rc<Accelerator>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}
