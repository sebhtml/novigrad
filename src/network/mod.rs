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
    pub layer_deltas: Vec<Tensor>,
    pub weight_deltas: Vec<Tensor>,
    pub addition: Tensor,
    pub w_t: Tensor,
    pub layer_f_derivative: Tensor,
    pub layer_delta: Tensor,
    pub layer_weight_delta: Tensor,
    pub output_diff: Tensor,
    pub next_layer_delta_transpose: Tensor,
    pub next_layer_weights_transpose: Tensor,
    pub output_diff_transpose: Tensor,
    pub previous_a_time_output_delta: Tensor,
    pub previous_action_t: Tensor,
    pub layer_weight_delta_transpose: Tensor,
    pub last_activation_row: Tensor,
    pub loss: Tensor,
    pub tmp: Tensor,
}

impl Default for TrainWorkingMemory {
    fn default() -> Self {
        Self {
            matrix_products: Default::default(),
            activation_tensors: Default::default(),
            layer_deltas: Default::default(),
            weight_deltas: Default::default(),
            addition: Default::default(),
            w_t: Default::default(),
            layer_f_derivative: Default::default(),
            layer_delta: Default::default(),
            layer_weight_delta: Default::default(),
            output_diff: Default::default(),
            next_layer_delta_transpose: Default::default(),
            next_layer_weights_transpose: Default::default(),
            output_diff_transpose: Default::default(),
            previous_a_time_output_delta: Default::default(),
            previous_action_t: Default::default(),
            layer_weight_delta_transpose: Default::default(),
            last_activation_row: Default::default(),
            loss: Default::default(),
            tmp: Default::default(),
        }
    }
}

pub struct PredictWorkingMemory {
    pub matrix_product: Tensor,
    pub last_activation_row: Tensor,
    pub w_t: Tensor,
    pub previous_activation_tensor: Tensor,
    pub activation_tensor: Tensor,
    pub activation_tensors: Vec<Tensor>,
}

impl Default for PredictWorkingMemory {
    fn default() -> Self {
        Self {
            matrix_product: Default::default(),
            last_activation_row: Default::default(),
            w_t: Default::default(),
            previous_activation_tensor: Default::default(),
            activation_tensor: Default::default(),
            activation_tensors: Default::default(),
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
        epoch: usize,
        inputs: &Vec<Tensor>,
        outputs: &Vec<Tensor>,
    ) {
        for i in 0..inputs.len() {
            self.train_back_propagation(working_memory, epoch, i, &inputs[i], &outputs[i]);
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
        let w_t = &mut working_memory.w_t;
        for i in 0..inputs.len() {
            self.predict(
                matrix_product,
                w_t,
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
        _epoch: usize,
        _example_index: usize,
        x: &Tensor,
        y: &Tensor,
    ) {
        let layers_count = self.layers.len();
        let learning_rate: f32 = 0.5;
        let matrix_products = &mut working_memory.matrix_products;
        matrix_products.resize_with(layers_count, Tensor::default);
        let activation_tensors = &mut working_memory.activation_tensors;
        activation_tensors.resize_with(layers_count, Tensor::default);

        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);

        let w_t = &mut working_memory.w_t;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let matrix_product = &mut matrix_products[layer_index];

            let previous_activation = {
                if layer_index == 0 {
                    &x
                } else {
                    let previous_layer_index = layer_index - 1;
                    &activation_tensors[previous_layer_index]
                }
            };

            // Use the same convention that is used in tensorflow:
            // y = x @ W^T+b
            // Weights is on the right.
            // W is transposed.
            // X is not transposed.
            let op_result = layer.forward(previous_activation, w_t, matrix_product);
            op_result.expect("Ok");
            let activation_function = &layer.activation();
            let activation_tensor = &mut activation_tensors[layer_index];
            let op_result = activation_function.activate(matrix_product, activation_tensor);
            op_result.expect("Ok");
        }

        let layer_deltas = &mut working_memory.layer_deltas;
        layer_deltas.resize_with(self.layers.len(), Tensor::default);

        let weight_deltas = &mut working_memory.weight_deltas;
        weight_deltas.resize_with(self.layers.len(), Tensor::default);

        let layer_f_derivative = &mut working_memory.layer_f_derivative;
        let layer_delta = &mut working_memory.layer_delta;
        let layer_weight_delta = &mut working_memory.layer_weight_delta;
        let output_diff = &mut working_memory.output_diff;
        let next_layer_delta_transpose = &mut working_memory.next_layer_delta_transpose;
        let next_layer_weights_transpose = &mut working_memory.next_layer_weights_transpose;
        let output_diff_transpose = &mut working_memory.output_diff_transpose;
        let previous_a_time_output_delta = &mut working_memory.previous_a_time_output_delta;
        let previous_action_t = &mut working_memory.previous_action_t;
        let layer_weight_delta_transpose = &mut working_memory.layer_weight_delta_transpose;

        // Back-propagation
        for (layer_index, _) in self.layers.iter().enumerate().rev() {
            let layer = &self.layers[layer_index];
            let layer_activation_function = &layer.activation();
            let layer_product_tensor = &matrix_products[layer_index];
            let layer_activation_tensor = &activation_tensors[layer_index];
            if layer_index == self.layers.len() - 1 && self.using_softmax_and_cross_entropy_loss {
                layer_activation_tensor
                    .scalar_add(1.0, layer_f_derivative)
                    .expect("Ok");
            } else {
                let op_result = layer_activation_function.derive(
                    layer_product_tensor,
                    layer_activation_tensor,
                    layer_f_derivative,
                );
                op_result.expect("Ok");
            }

            let binding = self.layers[layer_index].weights();
            let layer_weights: &Tensor = &binding.borrow();

            let previous_activation = {
                if layer_index == 0 {
                    &x
                } else {
                    let previous_layer_index = layer_index - 1;
                    &activation_tensors[previous_layer_index]
                }
            };

            if layer_index == self.layers.len() - 1 {
                // Output layer
                let last_activation_row = &mut working_memory.last_activation_row;
                let tmp = &mut working_memory.tmp;
                let loss = &mut working_memory.loss;
                let last_row = layer_activation_tensor.rows() - 1;
                layer_activation_tensor.row(last_row, last_activation_row);
                let op_result = self
                    .loss_function
                    .derive(tmp, y, &last_activation_row, loss);
                op_result.expect("Ok");
                //print_expected_output_and_actual_output(example_index, y, &clipped_tensor, Some(loss));
                //assert!(false);

                output_diff.reshape(
                    layer_activation_tensor.rows(),
                    layer_activation_tensor.cols(),
                );
                let mut col = 0;
                let cols = loss.cols();

                while col < cols {
                    let value = loss.get(0, col);
                    output_diff.set(last_row, col, value);
                    col += 1;
                }
            } else {
                // Hidden layer
                let next_layer_index = layer_index + 1;

                {
                    let next_layer_delta = &layer_deltas[next_layer_index];
                    next_layer_delta.transpose(next_layer_delta_transpose);
                }

                {
                    let binding = self.layers[next_layer_index].weights();
                    let next_layer_weights: &Tensor = &binding.borrow();
                    next_layer_weights.transpose(next_layer_weights_transpose);
                }
                let op_result =
                    next_layer_weights_transpose.matmul(next_layer_delta_transpose, output_diff);
                {
                    output_diff.transpose(output_diff_transpose);
                    swap(output_diff, output_diff_transpose);
                }

                op_result.expect("Ok");
            }

            let op_result = layer_f_derivative.element_wise_mul(output_diff, layer_delta);
            op_result.expect("Ok");

            previous_activation.transpose(previous_action_t);

            let op_result = previous_action_t.matmul(layer_delta, previous_a_time_output_delta);
            op_result.expect("Ok");
            let op_result =
                previous_a_time_output_delta.scalar_mul(learning_rate, layer_weight_delta);
            op_result.expect("Ok");

            {
                layer_weight_delta.transpose(layer_weight_delta_transpose);
                swap(layer_weight_delta, layer_weight_delta_transpose);
            }

            debug_assert_eq!(layer_weights.shape(), layer_weight_delta.shape());

            swap(&mut layer_deltas[layer_index], layer_delta);
            swap(&mut weight_deltas[layer_index], layer_weight_delta);
        }

        // Apply weight deltas
        let addition = &mut working_memory.addition;
        for layer in 0..self.layers.len() {
            {
                let binding = &self.layers[layer].weights();
                let weights: &Tensor = &binding.borrow();
                let op_result = weights.sub(&weight_deltas[layer], addition);
                op_result.expect("Ok");
            }
            let binding = &self.layers[layer].weights();
            let weights: &mut Tensor = &mut binding.borrow_mut();
            swap(weights, addition);
        }
    }

    pub fn predict_many(
        &self,
        matrix_product: &mut Tensor,
        w_t: &mut Tensor,
        previous_activation_tensor: &mut Tensor,
        inputs: &Vec<Tensor>,
        outputs: &mut Vec<Tensor>,
    ) {
        let len = inputs.len();
        let mut i = 0;
        outputs.resize_with(len, Tensor::default);
        while i < len {
            let input = &inputs[i];
            let activation_tensor = &mut outputs[i];
            self.predict(
                matrix_product,
                w_t,
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
        w_t: &mut Tensor,
        previous_activation_tensor: &mut Tensor,
        input: &Tensor,
        activation_tensor: &mut Tensor,
    ) {
        // Add a constant for bias
        //x.push(1.0);

        previous_activation_tensor.assign(input);
        for layer in self.layers.iter() {
            let op_result = layer.forward(previous_activation_tensor, w_t, matrix_product);
            op_result.expect("Ok");

            let activation_function = layer.activation();
            let op_result = activation_function.activate(&matrix_product, activation_tensor);
            op_result.expect("Ok");
            previous_activation_tensor.assign(activation_tensor);
        }
    }
}
