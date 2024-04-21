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
        _accelerator: &Accelerator,
        _inputs: &Vec<Rc<Tensor>>,
        _layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error> {
        Err(Error::UnsupportedOperation)
    }

    fn forward(
        &self,
        accelerator: &Accelerator,
        inputs: &Vec<Rc<Tensor>>,
    ) -> Result<Rc<Tensor>, Error>;

    fn backward2(
        &self,
        _inputs: &Vec<Rc<Tensor>>,
        _accelerator: &Accelerator,
        _layer_output_delta: &Tensor,
        _previous_layer_output_delta: &mut Tensor,
    ) {
    }

    fn backward(
        &self,
        accelerator: &Accelerator,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<Tensor>>,
        output: &Rc<Tensor>,
        back_propagated_delta: &mut Tensor,
        layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error> {
        self.get_layer_output_delta(
            accelerator,
            error_working_memory,
            inputs,
            output,
            back_propagated_delta,
            layer_delta,
        );

        let operator_gradients = self.compute_gradients(accelerator, inputs, layer_delta)?;
        self.backward2(inputs, accelerator, layer_delta, back_propagated_delta);

        Ok((back_propagated_delta.clone(), operator_gradients))
    }

    fn get_layer_output_delta(
        &self,
        _accelerator: &Accelerator,
        _working_memory: &mut DeltaWorkingMemory,
        _inputs: &Vec<Rc<Tensor>>,
        _output: &Rc<Tensor>,
        _back_propagated_layer_output_delta: &Tensor,
        _layer_output_delta: &mut Tensor,
    ) {
    }
}

pub trait Forward {
    fn forward(&mut self, input: &Rc<Tensor>) -> Result<Rc<Tensor>, Error>;
    fn accelerator(&self) -> Rc<Accelerator>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}
