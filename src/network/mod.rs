#[cfg(test)]
pub mod tests;
pub mod train;
use std::{cell::RefCell, mem::swap, rc::Rc};

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    loss::{LossFunction, LossFunctionName},
    Activation, ActivationFunction, Error, Layer, Linear, Tensor,
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
    pub activations: Vec<Tensor>,
    pub layer_deltas: Vec<Tensor>,
    pub weight_deltas: Vec<Tensor>,
    pub matrix_product: Tensor,
    pub addition: Tensor,
    pub w_t: Tensor,
    pub activation_tensor: Tensor,
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
}

impl Default for TrainWorkingMemory {
    fn default() -> Self {
        Self {
            matrix_products: Default::default(),
            activations: Default::default(),
            layer_deltas: Default::default(),
            weight_deltas: Default::default(),
            matrix_product: Default::default(),
            addition: Default::default(),
            w_t: Default::default(),
            activation_tensor: Default::default(),
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
        }
    }
}

impl Network {
    pub fn new(layer_configs: &Vec<LayerConfig>, loss_function_name: &LossFunctionName) -> Self {
        let mut rng = thread_rng();
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
                    let mut weights = Vec::new();
                    let rows = layer_config.rows;
                    let cols = layer_config.cols;
                    let right = (6.0 as f32).sqrt() / (cols as f32 + rows as f32).sqrt();
                    let left = -right;
                    // Xavier Initialization, or Glorot Initialization,
                    let uniform = Uniform::new(left, right);
                    weights.resize(rows * cols, 0.0);
                    for index in 0..weights.len() {
                        weights[index] = rng.sample(uniform);
                    }
                    let weights = Tensor::new(rows, cols, weights);
                    let activation: Rc<dyn ActivationFunction> =
                        layer_config.activation.clone().into();
                    Box::new(Linear {
                        weights: Rc::new(RefCell::new(weights)),
                        activation,
                    })
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

    pub fn total_error(&self, inputs: &Vec<Tensor>, outputs: &Vec<Tensor>) -> Result<f32, Error> {
        let mut total_error = 0.0;
        let mut predicted = Tensor::default();
        let mut last_activation_row = Tensor::default();
        for i in 0..inputs.len() {
            self.predict(&inputs[i], &mut predicted);
            let target = &outputs[i];
            let last_row = predicted.rows() - 1;
            predicted.row(last_row, &mut last_activation_row);
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
        let learning_rate: f32 = 0.5;
        let matrix_products = &mut working_memory.matrix_products;
        matrix_products.resize_with(0, Tensor::default);
        let activations = &mut working_memory.activations;
        activations.resize_with(0, Tensor::default);
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let matrix_product = &mut working_memory.matrix_product;
        let w_t = &mut working_memory.w_t;
        let activation_tensor = &mut working_memory.activation_tensor;

        //println!("input_shape {:?}", x.shape());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let previous_activation = {
                if layer_index == 0 {
                    &x
                } else {
                    &activations[activations.len() - 1]
                }
            };

            let activation_function = &layer.activation();
            // Use the same convention that is used in tensorflow:
            // y = x @ W^T+b
            // Weights is on the right.
            // W is transposed.
            // X is not transposed.
            let error = layer.forward(previous_activation, w_t, matrix_product);

            match error {
                Ok(_) => {
                    matrix_products.push(matrix_product.clone());
                    let op_result = activation_function.activate(matrix_product, activation_tensor);
                    op_result.expect("Ok");
                    //println!("layer_index {}  activation_shape {:?}", layer_index, activation_tensor.shape());
                    activations.push(activation_tensor.clone());
                }
                _ => {
                    let layer_weights = layer.weights();
                    (*layer_weights).borrow().transpose(w_t);
                    println!("In layer {}", layer_index);
                    println!("Incompatible shapes in matrix multiplication");
                    println!(
                        "Between X {:?} and W^T {:?}",
                        previous_activation.shape(),
                        w_t.shape(),
                    );
                    debug_assert!(false);
                }
            }
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
            let layer_activation_tensor = &activations[layer_index];
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
                    &activations[previous_layer_index]
                }
            };

            if layer_index == self.layers.len() - 1 {
                // Output layer
                let last_activation_row = &mut working_memory.last_activation_row;
                let loss = &mut working_memory.loss;
                let last_row = layer_activation_tensor.rows() - 1;
                layer_activation_tensor.row(last_row, last_activation_row);
                let op_result = self.loss_function.derive(y, &last_activation_row, loss);
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
                    *output_diff = output_diff_transpose.clone();
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
                *layer_weight_delta = layer_weight_delta_transpose.clone();
            }

            debug_assert_eq!(layer_weights.shape(), layer_weight_delta.shape());

            layer_deltas[layer_index] = layer_delta.clone();
            weight_deltas[layer_index] = layer_weight_delta.clone();
        }

        // Apply weight deltas
        let addition = &mut working_memory.addition;
        for layer in 0..self.layers.len() {
            let op_result = (*self.layers[layer].weights())
                .borrow()
                .sub(&weight_deltas[layer], addition);
            op_result.expect("Ok");
            *self.layers[layer].weights().as_ref().borrow_mut() = addition.clone();
        }
    }

    pub fn predict_many(&self, inputs: &Vec<Tensor>) -> Vec<Tensor> {
        let mut predicted = Tensor::default();
        let mut outputs = Vec::new();
        for input in inputs {
            self.predict(input, &mut predicted);
            outputs.push(predicted.clone());
        }
        outputs
    }

    pub fn predict(&self, x: &Tensor, predicted: &mut Tensor) {
        // Add a constant for bias
        //x.push(1.0);
        let mut matrix_product = Tensor::default();
        let mut w_t = Tensor::default();
        let mut activation_tensor = Tensor::default();

        predicted.assign(x);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let activation_function = layer.activation();
            let error = layer.forward(predicted, &mut w_t, &mut matrix_product);
            match error {
                Ok(_) => {
                    let op_result =
                        activation_function.activate(&matrix_product, &mut activation_tensor);
                    op_result.expect("Ok");
                    swap(predicted, &mut activation_tensor);
                }
                _ => {
                    let layer_weights = layer.weights();
                    (*layer_weights).borrow().transpose(&mut w_t);
                    println!("In layer {}", layer_index);
                    println!("Incompatible shapes in matrix multiplication");
                    println!(
                        "Between X {:?} and W^T {:?}",
                        predicted.shape(),
                        w_t.shape(),
                    );
                    debug_assert!(false);
                }
            }
        }
    }
}
