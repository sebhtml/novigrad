#[cfg(test)]
pub mod tests;
mod train;
use std::{cell::RefCell, ops::Deref, rc::Rc};
pub use train::*;

use crate::{
    accelerator::Accelerator,
    back_propagation,
    loss::{LossFunction, LossFunctionType},
    Error, Forward, Optimizer, OptimizerTrait, Tape, Tensor,
};

pub struct Network {
    architecture: Box<dyn Forward>,
    loss_function: LossFunctionType,
    accelerator: Rc<Accelerator>,
    optimizer: Optimizer,
    tape: Rc<RefCell<Tape>>,
}

pub struct TrainWorkingMemory {
    pub layer_output: Tensor,
    pub next_layer_delta: Tensor,
    pub back_propagated_delta: Tensor,
    pub layer_delta: Tensor,
    pub previous_activation_tensor: Tensor,
    pub tmp: Tensor,
}

impl Default for TrainWorkingMemory {
    fn default() -> Self {
        Self {
            layer_output: Default::default(),
            next_layer_delta: Default::default(),
            back_propagated_delta: Default::default(),
            layer_delta: Default::default(),
            previous_activation_tensor: Default::default(),
            tmp: Default::default(),
        }
    }
}

pub struct DeltaWorkingMemory {
    pub layer_f_derivative: Tensor,
}

impl Default for DeltaWorkingMemory {
    fn default() -> Self {
        Self {
            layer_f_derivative: Default::default(),
        }
    }
}

pub struct PredictWorkingMemory {
    pub previous_activation_tensor: Tensor,
    pub activation_tensor: Tensor,
    pub activation_tensors: Vec<Tensor>,
}

impl PredictWorkingMemory {
    pub fn new(examples_count: usize) -> Self {
        Self {
            previous_activation_tensor: Default::default(),
            activation_tensor: Default::default(),
            activation_tensors: vec![Tensor::default(); examples_count],
        }
    }
}

impl Network {
    pub fn new(architecture: Box<dyn Forward>, loss_function: LossFunctionType) -> Self {
        let accelerator = architecture.accelerator();
        let tape = architecture.tape();
        Self {
            architecture,
            loss_function,
            accelerator,
            tape,
            optimizer: Default::default(),
        }
    }

    pub fn train(
        &mut self,
        working_memory: &mut TrainWorkingMemory,
        error_working_memory: &mut DeltaWorkingMemory,
        epoch: usize,
        inputs: &Vec<Tensor>,
        outputs: &Vec<Tensor>,
    ) -> Result<(), Error> {
        for i in 0..inputs.len() {
            self.train_back_propagation(
                working_memory,
                error_working_memory,
                epoch,
                i,
                &inputs[i],
                &outputs[i],
            )?;
        }
        Ok(())
    }

    pub fn total_error(
        &mut self,
        working_memory: &mut PredictWorkingMemory,
        inputs: &Vec<Tensor>,
        outputs: &Vec<Tensor>,
    ) -> Result<f32, Error> {
        let mut total_error = 0.0;
        let activation_tensor = &mut working_memory.activation_tensor;
        let previous_activation_tensor = &mut working_memory.previous_activation_tensor;

        for i in 0..inputs.len() {
            self.forward(previous_activation_tensor, &inputs[i], activation_tensor)?;
            let target = &outputs[i];
            let example_error = self
                .loss_function
                .evaluate(&self.accelerator, target, &activation_tensor)
                .expect("Ok");
            total_error += example_error;
        }

        Ok(total_error)
    }

    fn train_back_propagation(
        &mut self,
        working_memory: &mut TrainWorkingMemory,
        error_working_memory: &mut DeltaWorkingMemory,
        _epoch: usize,
        _example_index: usize,
        x: &Tensor,
        y: &Tensor,
    ) -> Result<(), Error> {
        self.tape.deref().borrow_mut().clear();

        {
            let layer_output = &mut working_memory.layer_output;
            let previous_activation_tensor = &mut working_memory.previous_activation_tensor;
            self.forward(previous_activation_tensor, x, layer_output)?;
        }

        back_propagation(
            x,
            y,
            working_memory,
            error_working_memory,
            &self.loss_function,
            &self.accelerator,
            &self.tape,
        );

        self.optimizer.optimize(&self.tape, &self.accelerator);
        Ok(())
    }

    pub fn predict_many(
        &mut self,
        previous_activation_tensor: &mut Tensor,
        inputs: &Vec<Tensor>,
        activation_tensors: &mut Vec<Tensor>,
    ) -> Result<(), Error> {
        let len = inputs.len();
        let mut i = 0;
        while i < len {
            let input = &inputs[i];
            let activation_tensor = &mut activation_tensors[i];
            self.forward(previous_activation_tensor, input, activation_tensor)?;
            i += 1;
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        _previous_activation_tensor: &mut Tensor,
        input: &Tensor,
        activation_tensor: &mut Tensor,
    ) -> Result<(), Error> {
        let output = self.architecture.forward(input)?;
        activation_tensor.assign(&self.accelerator, &output);
        Ok(())
    }
}
