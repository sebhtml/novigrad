use std::{borrow::Borrow, cell::RefCell, rc::Rc};

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
                    let right = ((6.0 as f32).sqrt() / (cols as f32 + rows as f32).sqrt());
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

    fn train_back_propagation(&mut self, epoch: usize, _example: usize, x: &Tensor, y: &Tensor) {
        let learning_rate: f32 = 0.5;
        //println!("Learning rate {}", learning_rate);
        let x = x;
        let y = y;
        let mut matrix_products: Vec<Tensor> = Vec::new();
        let mut activations: Vec<Tensor> = Vec::new();
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let mut matrix_product = Tensor::default();
        let mut addition = Tensor::default();

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let previous_activation = {
                if layer_index == 0 {
                    &x
                } else {
                    &activations[activations.len() - 1]
                }
            };

            let activation = layer.activation();
            // Use the same convention that is used in tensorflow:
            //  y= x W^T+b
            // Weights is on the right.
            // W is transposed.
            // X is not transposed.
            let error = layer.forward(&previous_activation, &mut matrix_product);

            match error {
                Ok(_) => {
                    /*
                                       println!("Forward Layer {}", layer_index);
                                       println!("previous_activation {}", previous_activation);
                                       println!("weights^T {}", (*layer.weights()).borrow().transpose());
                                       println!("matrix_product {}", matrix_product);
                    */
                    matrix_products.push(matrix_product.clone());
                    let activation = activation.activate_matrix(matrix_product.clone());
                    activations.push(activation);
                }
                _ => {
                    let layer_weights = layer.weights();
                    println!("Incompatible shapes in matrix multiplication");
                    println!(
                        "Between  X {} and W {}",
                        previous_activation,
                        (*layer_weights).borrow().clone().transpose(),
                    );
                }
            }
        }

        let mut layer_deltas = Vec::new();
        layer_deltas.resize(self.layers.len(), Tensor::default());

        let mut weight_deltas: Vec<Tensor> = Vec::new();
        weight_deltas.resize(self.layers.len(), Tensor::default());

        // Back-propagation
        for (layer_index, _) in self.layers.iter().enumerate().rev() {
            let layer = &self.layers[layer_index];
            let activation_function = &layer.activation();
            let activation_tensor = &activations[layer_index];

            assert_eq!(layer_index, self.layers.len() - 1);

            if layer_index == self.layers.len() - 1 {
                /*
                               println!("expected_tensor {}", y);
                               println!("activation_tensor {}", activation_tensor);
                */
                let f_derivative = activation_function.derive_matrix(activation_tensor.clone());
                let mut output_diff = Tensor::default();
                let op_result = y.sub(&activation_tensor, &mut output_diff);
                op_result.expect("Ok");
                let mut layer_delta = Tensor::default();
                let op_result = f_derivative.element_wise_mul(&output_diff, &mut layer_delta);
                op_result.expect("Ok");
                /*
                               println!("f_derivative {}", f_derivative);
                               println!("output_diff {}", output_diff);
                               println!("layer_delta {}", layer_delta);
                */
                let weights = &layer.weights();
                //println!("weights^T {}", (**weights).borrow().clone().transpose());

                let previous_activation = &x;
                //println!("previous_activation {}", previous_activation);

                let mut previous_a_time_output_delta = Tensor::default();
                let previous_action_t = previous_activation.transpose();
                let op_result =
                    previous_action_t.matmul(&layer_delta, &mut previous_a_time_output_delta);
                op_result.expect("Ok");
                let mut weight_delta = Tensor::default();
                let op_result =
                    previous_a_time_output_delta.scalar_mul(learning_rate, &mut weight_delta);
                op_result.expect("Ok");
                //println!("weight_delta {}", weight_delta);

                //assert!(false);

                layer_deltas[layer_index] = layer_delta;
                weight_deltas[layer_index] = weight_delta.transpose();
            }
        }

        // Apply deltas
        for layer in 0..self.layers.len() {
            /*
            println!("Weight delta {}", weight_deltas[layer]);
            println!(
                "Weights before {}",
                (*self.layers[layer].weights()).borrow()
            );
            */
            let op_result = (*self.layers[layer].weights())
                .borrow()
                .add(&weight_deltas[layer], &mut addition);
            op_result.expect("Ok");
            *self.layers[layer].weights().as_ref().borrow_mut() = addition.clone();
            //println!("Weights after {}", (*self.layers[layer].weights()).borrow());
        }
    }

    fn col_diff(&self, y: &Tensor, output: &Tensor, col: usize) -> f32 {
        // TODO we should check the last row
        let diff = y.get(0, col) - output.get(0, col);
        diff
    }

    fn compute_error(&self, y: &Tensor, output: &Tensor) -> f32 {
        let mut error = 0.0;
        for col in 0..y.cols() {
            let diff = self.col_diff(y, output, col);
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
        let mut previous_activation = x.clone();
        let mut matrix_product = Tensor::default();

        for layer in self.layers.iter() {
            let activation = layer.activation();
            let error = layer.forward(&previous_activation, &mut matrix_product);
            match error {
                Ok(_) => {
                    let activation = activation.activate_matrix(matrix_product.clone());
                    previous_activation = activation;
                }
                _ => {
                    let layer_weights = layer.weights();
                    println!("Incompatible shapes in matrix multiplication");
                    println!(
                        "Between  X {} and W {}",
                        previous_activation,
                        (*layer_weights).borrow().clone().transpose(),
                    );
                }
            }
        }

        previous_activation
    }
}
