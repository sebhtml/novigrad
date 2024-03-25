pub mod tests;
use std::{cell::RefCell, rc::Rc};

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{Activation, ActivationFunction, Layer, Linear, Tensor};

pub struct LayerConfig {
    pub rows: usize,
    pub cols: usize,
    pub activation: Activation,
}

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(layer_configs: Vec<LayerConfig>) -> Self {
        let mut rng = thread_rng();

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
                    let activation = layer_config.activation;
                    weights.resize(rows * cols, 0.0);
                    for index in 0..weights.len() {
                        weights[index] = rng.sample(uniform);
                    }
                    let weights = Tensor::new(rows, cols, weights);
                    let activation: Rc<dyn ActivationFunction> = activation.into();
                    Box::new(Linear {
                        weights: Rc::new(RefCell::new(weights)),
                        activation,
                    })
                })
                .collect(),
        }
    }

    pub fn train(&mut self, epoch: usize, inputs: &Vec<Tensor>, outputs: &Vec<Tensor>) {
        for i in 0..inputs.len() {
            self.train_back_propagation(epoch, i, &inputs[i], &outputs[i]);
        }
    }

    pub fn total_error(&self, inputs: &Vec<Tensor>, outputs: &Vec<Tensor>) -> f32 {
        let mut total_error = 0.0;
        for i in 0..inputs.len() {
            let predicted = self.predict(&inputs[i]);
            let target = &outputs[i];
            let example_error = self.compute_error(target, &predicted);
            total_error += example_error;
        }

        total_error
    }

    fn train_back_propagation(&mut self, _epoch: usize, _example: usize, x: &Tensor, y: &Tensor) {
        let learning_rate: f32 = 0.5;
        let mut matrix_products: Vec<Tensor> = Vec::new();
        let mut activations: Vec<Tensor> = Vec::new();
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let mut matrix_product = Tensor::default();
        let mut addition = Tensor::default();
        let mut w_t = Tensor::default();
        let mut activation_tensor = Tensor::default();

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
            let error = layer.forward(&previous_activation, &mut w_t, &mut matrix_product);

            match error {
                Ok(_) => {
                    matrix_products.push(matrix_product.clone());
                    let op_result =
                        activation_function.activate(&matrix_product, &mut activation_tensor);
                    op_result.expect("Ok");
                    activations.push(activation_tensor.clone());
                }
                _ => {
                    let layer_weights = layer.weights();
                    (*layer_weights).borrow().transpose(&mut w_t);
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

        let mut layer_deltas = Vec::new();
        layer_deltas.resize(self.layers.len(), Tensor::default());

        let mut weight_deltas: Vec<Tensor> = Vec::new();
        weight_deltas.resize(self.layers.len(), Tensor::default());

        let mut layer_f_derivative = Tensor::default();
        let mut layer_delta = Tensor::default();
        let mut layer_weight_delta = Tensor::default();
        let mut output_diff = Tensor::default();
        let mut next_layer_delta_transpose = Tensor::default();
        let mut next_layer_weights_transpose = Tensor::default();
        let mut output_diff_transpose = Tensor::default();
        let mut previous_a_time_output_delta = Tensor::default();
        let mut previous_action_t = Tensor::default();
        let mut layer_weight_delta_transpose = Tensor::default();

        // Back-propagation
        for (layer_index, _) in self.layers.iter().enumerate().rev() {
            let layer = &self.layers[layer_index];
            let layer_activation_function = &layer.activation();
            let layer_product_tensor = &matrix_products[layer_index];
            let layer_activation_tensor = &activations[layer_index];
            let op_result = layer_activation_function.derive(
                &layer_product_tensor,
                &layer_activation_tensor,
                &mut layer_f_derivative,
            );
            op_result.expect("Ok");
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
                debug_assert_eq!(y.cols(), layer_activation_tensor.cols());
                let op_result = y.sub_broadcast(&layer_activation_tensor, &mut output_diff);
                op_result.expect("Ok");
            } else {
                // Hidden layer
                let next_layer_index = layer_index + 1;

                {
                    let next_layer_delta = &layer_deltas[next_layer_index];
                    next_layer_delta.transpose(&mut next_layer_delta_transpose);
                }

                {
                    let binding = self.layers[next_layer_index].weights();
                    let next_layer_weights: &Tensor = &binding.borrow();
                    next_layer_weights.transpose(&mut next_layer_weights_transpose);
                }
                let op_result = next_layer_weights_transpose
                    .matmul(&next_layer_delta_transpose, &mut output_diff);
                {
                    output_diff.transpose(&mut output_diff_transpose);
                    output_diff = output_diff_transpose.clone();
                }

                op_result.expect("Ok");
            }

            let op_result = layer_f_derivative.element_wise_mul(&output_diff, &mut layer_delta);
            op_result.expect("Ok");

            previous_activation.transpose(&mut previous_action_t);

            let op_result =
                previous_action_t.matmul(&layer_delta, &mut previous_a_time_output_delta);
            op_result.expect("Ok");
            let op_result =
                previous_a_time_output_delta.scalar_mul(learning_rate, &mut layer_weight_delta);
            op_result.expect("Ok");

            {
                layer_weight_delta.transpose(&mut layer_weight_delta_transpose);
                layer_weight_delta = layer_weight_delta_transpose.clone();
            }

            debug_assert_eq!(layer_weights.shape(), layer_weight_delta.shape());

            layer_deltas[layer_index] = layer_delta.clone();
            weight_deltas[layer_index] = layer_weight_delta.clone();
        }

        // Apply weight deltas
        for layer in 0..self.layers.len() {
            let op_result = (*self.layers[layer].weights())
                .borrow()
                .add(&weight_deltas[layer], &mut addition);
            op_result.expect("Ok");
            *self.layers[layer].weights().as_ref().borrow_mut() = addition.clone();
        }
    }

    fn compute_error(&self, y: &Tensor, output: &Tensor) -> f32 {
        let mut error = 0.0;
        let last_row = output.rows() - 1;
        debug_assert_eq!(output.cols(), y.cols());
        for col in 0..y.cols() {
            let diff = y.get(0, col) - output.get(last_row, col);
            error += diff.powf(2.0);
        }
        error * 0.5
    }

    pub fn predict_many(&self, inputs: &Vec<Tensor>) -> Vec<Tensor> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        // Add a constant for bias
        //x.push(1.0);
        let mut previous_activation = Tensor::default();
        let mut matrix_product = Tensor::default();
        let mut w_t = Tensor::default();

        previous_activation = x.clone();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let activation_function = layer.activation();
            let error = layer.forward(&previous_activation, &mut w_t, &mut matrix_product);
            match error {
                Ok(_) => {
                    let mut activation_tensor = Tensor::default();
                    let op_result =
                        activation_function.activate(&matrix_product, &mut activation_tensor);
                    op_result.expect("Ok");
                    previous_activation = activation_tensor;
                }
                _ => {
                    let layer_weights = layer.weights();
                    (*layer_weights).borrow().transpose(&mut w_t);
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

        previous_activation
    }
}
