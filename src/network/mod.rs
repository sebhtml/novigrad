#[cfg(test)]
pub mod tests;
mod train;
use std::mem::swap;
pub use train::*;

use crate::{
    accelerator::Accelerator,
    loss::{LossFunction, LossFunctionType},
    DifferentiableModule, DifferentiableModuleConfig, DifferentiableModuleTrait, Error, Tensor,
};

pub struct Network<'a> {
    layers: Vec<DifferentiableModule>,
    loss_function: &'a LossFunctionType,
    accelerator: Accelerator,
}

pub struct TrainWorkingMemory {
    pub layer_outputs: Vec<Tensor>,
    pub next_layer_delta: Tensor,
    pub back_propagated_delta: Tensor,
    pub layer_delta: Tensor,
    pub previous_activation_tensor: Tensor,
    pub tmp: Tensor,
}

impl TrainWorkingMemory {
    pub fn new(layers: usize) -> Self {
        Self {
            layer_outputs: vec![Default::default(); layers],
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

impl<'a> Network<'a> {
    pub fn new(
        layer_configs: &Vec<DifferentiableModuleConfig>,
        loss_function: &'a LossFunctionType,
    ) -> Self {
        Self {
            layers: layer_configs
                .into_iter()
                .map(|layer_config| layer_config.into())
                .collect(),
            loss_function,
            accelerator: Default::default(),
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
            self.predict(previous_activation_tensor, &inputs[i], activation_tensor);
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
        let learning_rate: f32 = 0.5;
        let layer_outputs = &mut working_memory.layer_outputs;

        for layer_index in 0..self.layers.len() {
            let previous_activation_tensor = &mut working_memory.previous_activation_tensor;
            if layer_index == 0 {
                previous_activation_tensor.assign(&self.accelerator, x);
            }
            let layer_output = &mut layer_outputs[layer_index];

            let layer = &mut self.layers[layer_index];
            let op_result =
                layer.forward(&self.accelerator, previous_activation_tensor, layer_output);
            op_result.expect("Ok");
            previous_activation_tensor.assign(&self.accelerator, &layer_output);
        }

        let next_layer_delta = &mut working_memory.next_layer_delta;
        let layer_delta = &mut working_memory.layer_delta;
        let layers_count = self.layers.len();

        // Back-propagation
        for layer_index in (0..layers_count).into_iter().rev() {
            let is_last_layer = layer_index == self.layers.len() - 1;

            let previous_activation_tensor = match layer_index {
                0 => x,
                _ => &layer_outputs[layer_index - 1],
            };

            if is_last_layer {
                // For the output layer, the next layer delta is the loss.
                let layer_activation_tensor = &layer_outputs[layer_index];

                let op_result = self.loss_function.derive(
                    &self.accelerator,
                    y,
                    &layer_activation_tensor,
                    next_layer_delta,
                );
                op_result.expect("Ok");
            }

            {
                let next_layer = if is_last_layer {
                    None
                } else {
                    let next_layer_index = layer_index + 1;
                    Some(&self.layers[next_layer_index])
                };

                let layer = &self.layers[layer_index];
                let tmp = &mut working_memory.tmp;
                let layer_input = previous_activation_tensor;
                let layer_output = &layer_outputs[layer_index];
                let back_propagated_delta = &mut working_memory.back_propagated_delta;

                let is_last_layer = next_layer.is_none();
                match next_layer {
                    None => {
                        // use the output of the loss function¸
                        back_propagated_delta.assign(&self.accelerator, next_layer_delta);
                    }
                    Some(next_layer) => {
                        // Hidden layer
                        next_layer.backward(
                            &self.accelerator,
                            next_layer_delta,
                            back_propagated_delta,
                        );
                    }
                }

                layer.get_layer_output_delta(
                    &self.accelerator,
                    error_working_memory,
                    layer_input,
                    layer_output,
                    back_propagated_delta,
                    is_last_layer,
                    tmp,
                );

                tmp.clip(-1.0, 1.0, layer_delta)
            }

            {
                let layer = &mut self.layers[layer_index];
                layer.compute_gradient(&self.accelerator, previous_activation_tensor, layer_delta);
            }

            swap(next_layer_delta, layer_delta);
        }

        // Apply changes
        for layer in 0..self.layers.len() {
            let op_result = self.layers[layer].commit_change(&self.accelerator, learning_rate);
            op_result.expect("Ok");
        }
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
            self.predict(previous_activation_tensor, input, activation_tensor);
            i += 1;
        }
    }

    pub fn predict(
        &mut self,
        previous_activation_tensor: &mut Tensor,
        input: &Tensor,
        activation_tensor: &mut Tensor,
    ) {
        previous_activation_tensor.assign(&self.accelerator, input);
        for layer_index in 0..self.layers.len() {
            let layer = &mut self.layers[layer_index];
            let op_result = layer.forward(
                &self.accelerator,
                previous_activation_tensor,
                activation_tensor,
            );
            op_result.expect("Ok");
            previous_activation_tensor.assign(&self.accelerator, activation_tensor);
        }
    }
}
