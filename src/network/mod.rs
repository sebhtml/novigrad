#[cfg(test)]
pub mod tests;
mod train;
use std::{borrow::Borrow, cell::RefCell, ops::Deref, rc::Rc};
pub use train::*;

use crate::{
    accelerator::Accelerator,
    back_propagation,
    loss::{LossFunction, LossFunctionType},
    DifferentiableModule, DifferentiableModuleConfig, Error, FullDifferentiableModuleConfig,
    Optimizer, OptimizerTrait, Tape, Tensor,
};

pub struct Network {
    forward_layers: Vec<DifferentiableModule>,
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
    pub fn new(
        layer_configs: &Vec<DifferentiableModuleConfig>,
        loss_function: LossFunctionType,
    ) -> Self {
        let accelerator = Rc::new(Default::default());
        let tape = Rc::new(RefCell::new(Default::default()));
        Self {
            forward_layers: layer_configs
                .into_iter()
                .map(|layer_config| {
                    FullDifferentiableModuleConfig {
                        accelerator: &accelerator,
                        tape: &tape,
                        config: &layer_config,
                    }
                    .borrow()
                    .into()
                })
                .collect(),
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
    ) {
        for i in 0..inputs.len() {
            self.train_back_propagation(
                working_memory,
                error_working_memory,
                epoch,
                i,
                &inputs[i],
                &outputs[i],
            );
        }
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
            self.forward(previous_activation_tensor, &inputs[i], activation_tensor);
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
    ) {
        self.tape.deref().borrow_mut().clear();

        {
            let layer_output = &mut working_memory.layer_output;
            let previous_activation_tensor = &mut working_memory.previous_activation_tensor;
            self.forward(previous_activation_tensor, x, layer_output);
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
    }

    pub fn predict_many(
        &mut self,
        previous_activation_tensor: &mut Tensor,
        inputs: &Vec<Tensor>,
        activation_tensors: &mut Vec<Tensor>,
    ) {
        let len = inputs.len();
        let mut i = 0;
        while i < len {
            let input = &inputs[i];
            let activation_tensor = &mut activation_tensors[i];
            self.forward(previous_activation_tensor, input, activation_tensor);
            i += 1;
        }
    }

    pub fn forward(
        &mut self,
        previous_activation_tensor: &mut Tensor,
        input: &Tensor,
        activation_tensor: &mut Tensor,
    ) {
        previous_activation_tensor.assign(&self.accelerator, input);
        for layer_index in 0..self.forward_layers.len() {
            let layer = &mut self.forward_layers[layer_index];
            let op_result = layer.forward(previous_activation_tensor, activation_tensor);
            op_result.expect("Ok");
            previous_activation_tensor.assign(&self.accelerator, activation_tensor);
        }
    }
}
