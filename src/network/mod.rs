#[cfg(test)]
pub mod tests;
pub mod train;
use std::mem::swap;

use crate::{
    loss::{LossFunction, LossFunctionName},
    Activation, Error, Layer, Linear, Tensor,
};

pub struct LayerConfig {
    pub rows: usize,
    pub cols: usize,
    pub activation: Activation,
}

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    loss_function: Box<dyn LossFunction>,
    using_softmax_and_cross_entropy_loss: bool,
}

pub struct TrainWorkingMemory {
    pub matrix_products: Vec<Tensor>,
    pub activation_tensors: Vec<Tensor>,
    pub next_layer_delta: Tensor,
    pub addition: Tensor,
    pub layer_delta: Tensor,
    pub previous_activation_tensor: Tensor,
    pub last_activation_row: Tensor,
    pub loss: Tensor,
    pub tmp: Tensor,
}

impl TrainWorkingMemory {
    pub fn new(layers_count: usize) -> Self {
        Self {
            matrix_products: vec![Tensor::default(); layers_count],
            activation_tensors: vec![Tensor::default(); layers_count],
            next_layer_delta: Default::default(),
            addition: Default::default(),
            layer_delta: Default::default(),
            previous_activation_tensor: Default::default(),
            last_activation_row: Default::default(),
            loss: Default::default(),
            tmp: Default::default(),
        }
    }
}

pub struct DeltaWorkingMemory {
    pub output_diff: Tensor,
    pub layer_f_derivative: Tensor,
}

impl Default for DeltaWorkingMemory {
    fn default() -> Self {
        Self {
            output_diff: Default::default(),
            layer_f_derivative: Default::default(),
        }
    }
}

pub struct PredictWorkingMemory {
    pub matrix_product: Tensor,
    pub last_activation_row: Tensor,
    pub previous_activation_tensor: Tensor,
    pub activation_tensor: Tensor,
    pub activation_tensors: Vec<Tensor>,
}

impl PredictWorkingMemory {
    pub fn new(examples_count: usize) -> Self {
        Self {
            matrix_product: Default::default(),
            last_activation_row: Default::default(),
            previous_activation_tensor: Default::default(),
            activation_tensor: Default::default(),
            activation_tensors: vec![Tensor::default(); examples_count],
        }
    }
}

impl Network {
    pub fn new(layer_configs: &Vec<LayerConfig>, loss_function_name: &LossFunctionName) -> Self {
        let mut using_softmax_and_cross_entropy_loss = false;
        if loss_function_name == &LossFunctionName::CrossEntropyLoss {
            match layer_configs.last() {
                Some(config) => {
                    if config.activation != Activation::Softmax {
                        assert!(false, "CrossEntropyLoss only works with Softmax");
                    } else {
                        using_softmax_and_cross_entropy_loss = true;
                    }
                }
                _ => (),
            }
        }
        Self {
            layers: layer_configs
                .into_iter()
                .map(|layer_config| -> Box<dyn Layer> {
                    let rows = layer_config.rows;
                    let cols = layer_config.cols;
                    Box::new(Linear::new(
                        rows,
                        cols,
                        layer_config.activation.clone().into(),
                    ))
                })
                .collect(),
            loss_function: loss_function_name.into(),
            using_softmax_and_cross_entropy_loss,
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
        &self,
        working_memory: &mut PredictWorkingMemory,
        inputs: &Vec<Tensor>,
        outputs: &Vec<Tensor>,
    ) -> Result<f32, Error> {
        let mut total_error = 0.0;
        let activation_tensor = &mut working_memory.activation_tensor;
        let previous_activation_tensor = &mut working_memory.previous_activation_tensor;
        let last_activation_row = &mut working_memory.last_activation_row;
        let matrix_product = &mut working_memory.matrix_product;
        for i in 0..inputs.len() {
            self.predict(
                matrix_product,
                previous_activation_tensor,
                &inputs[i],
                activation_tensor,
            );
            let target = &outputs[i];
            let last_row = activation_tensor.rows() - 1;
            activation_tensor.row(last_row, last_activation_row);
            let example_error = self.loss_function.evaluate(target, &last_activation_row)?;
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
        let matrix_products = &mut working_memory.matrix_products;
        let activation_tensors = &mut working_memory.activation_tensors;

        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let matrix_product = &mut matrix_products[layer_index];

            let previous_activation_tensor = &mut working_memory.previous_activation_tensor;
            if layer_index == 0 {
                previous_activation_tensor.assign(x);
            } else {
                let previous_layer_index = layer_index - 1;
                previous_activation_tensor.assign(&activation_tensors[previous_layer_index]);
            }

            // Use the same convention that is used in tensorflow:
            // y = x @ W^T+b
            // Weights is on the right.
            // W is transposed.
            // X is not transposed.
            let activation_tensor = &mut activation_tensors[layer_index];
            let op_result = layer.forward(
                previous_activation_tensor,
                matrix_product,
                activation_tensor,
            );
            op_result.expect("Ok");
        }

        let next_layer_delta = &mut working_memory.next_layer_delta;
        let layer_delta = &mut working_memory.layer_delta;
        let layers_count = self.layers.len();

        // Back-propagation
        for layer_index in (0..layers_count).into_iter().rev() {
            let is_first_layer = layer_index == 0;
            let is_last_layer = layer_index == self.layers.len() - 1;
            let layer_product_tensor = &matrix_products[layer_index];
            let layer_activation_tensor = &activation_tensors[layer_index];

            let previous_activation = {
                if is_first_layer {
                    &x
                } else {
                    let previous_layer_index = layer_index - 1;
                    &activation_tensors[previous_layer_index]
                }
            };

            if is_last_layer {
                // For the output layer, the next layer delta is the loss.
                let last_activation_row = &mut working_memory.last_activation_row;
                let tmp = &mut working_memory.tmp;
                let loss = &mut working_memory.loss;
                let last_row = layer_activation_tensor.rows() - 1;
                layer_activation_tensor.row(last_row, last_activation_row);
                let op_result = self
                    .loss_function
                    .derive(tmp, y, &last_activation_row, loss);
                op_result.expect("Ok");

                next_layer_delta.reshape(
                    layer_activation_tensor.rows(),
                    layer_activation_tensor.cols(),
                );
                let mut col = 0;
                let cols = loss.cols();

                while col < cols {
                    let value = loss.get(0, col);
                    next_layer_delta.set(last_row, col, value);
                    col += 1;
                }
            }

            {
                let next_layer = if is_last_layer {
                    None
                } else {
                    let next_layer_index = layer_index + 1;
                    Some(&self.layers[next_layer_index])
                };

                let layer = &self.layers[layer_index];
                layer.get_layer_delta(
                    error_working_memory,
                    layer_product_tensor,
                    layer_activation_tensor,
                    next_layer,
                    next_layer_delta,
                    self.using_softmax_and_cross_entropy_loss,
                    layer_delta,
                );
            }

            {
                let layer = &mut self.layers[layer_index];
                layer.plan_change(learning_rate, previous_activation, layer_delta);
            }

            swap(next_layer_delta, layer_delta);
        }

        // Apply weight deltas
        for layer in 0..self.layers.len() {
            let op_result = self.layers[layer].commit_change();
            op_result.expect("Ok");
        }
    }

    pub fn predict_many(
        &self,
        matrix_product: &mut Tensor,
        previous_activation_tensor: &mut Tensor,
        inputs: &Vec<Tensor>,
        activation_tensors: &mut Vec<Tensor>,
    ) {
        let len = inputs.len();
        let mut i = 0;
        while i < len {
            let input = &inputs[i];
            let activation_tensor = &mut activation_tensors[i];
            self.predict(
                matrix_product,
                previous_activation_tensor,
                input,
                activation_tensor,
            );
            i += 1;
        }
    }

    pub fn predict(
        &self,
        matrix_product: &mut Tensor,
        previous_activation_tensor: &mut Tensor,
        input: &Tensor,
        activation_tensor: &mut Tensor,
    ) {
        // Add a constant for bias
        //x.push(1.0);

        previous_activation_tensor.assign(input);
        for layer in self.layers.iter() {
            let op_result = layer.forward(
                previous_activation_tensor,
                matrix_product,
                activation_tensor,
            );
            op_result.expect("Ok");
            previous_activation_tensor.assign(activation_tensor);
        }
    }
}
