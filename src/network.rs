use std::{cell::RefCell, rc::Rc};

use rand::Rng;

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
        Self {
            layers: layer_configs
                .into_iter()
                .map(|layer_config| -> Box<dyn Layer> {
                    let mut weights = Vec::new();
                    let rows = layer_config.rows;
                    let cols = layer_config.cols;
                    let activation = layer_config.activation;
                    weights.resize(rows * cols, 0.0);
                    for index in 0..weights.len() {
                        weights[index] = rand::thread_rng().gen_range(0.0..1.0);
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

    pub fn train(&mut self, inputs: &Vec<Tensor>, outputs: &Vec<Tensor>) {
        for i in 0..inputs.len() {
            self.train_back_propagation(i, &inputs[i], &outputs[i]);
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

    fn train_back_propagation(&mut self, _example: usize, x: &Tensor, y: &Tensor) {
        let learning_rate = 0.5;
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
                                        println!("Layer {}", layer_index);
                                        println!("previous_activation {}", previous_activation);
                                        println!("weights^T {}", layer.weights().borrow().transpose());
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
                        *layer_weights.borrow(),
                    );
                }
            }
        }

        // Back-propagation
        let mut weight_deltas: Vec<Tensor> = self
            .layers
            .iter()
            .map(|x| x.weights().as_ref().borrow().clone())
            .collect();
        let mut layer_diffs = Vec::new();
        layer_diffs.resize(self.layers.len(), Tensor::default());

        for (layer, _) in self.layers.iter().enumerate().rev() {
            let layer = layer.to_owned();
            let activation = self.layers[layer].activation();
            let layer_activation = &activations[layer];
            let derived_matrix = activation.derive_matrix(layer_activation.clone());
            //println!("layer {}, layer_activation {}", layer, layer_activation);
            //println!("layer {}, derived_matrix {}", layer, derived_matrix);
            layer_diffs[layer].reshape(layer_activation.rows(), layer_activation.cols());
            let activation_rows = layer_activation.rows();
            let activation_cols = layer_activation.cols();
            for row in 0..activation_rows {
                for col in 0..activation_cols {
                    // For that activation value, how far off are we ?
                    let target_diff = if layer == self.layers.len() - 1 {
                        // If it's the final layer, compare with example Y.
                        let diff = y.get(0, col) - layer_activation.get(row, col);
                        /*
                        println!("Output row {}", row);
                        println!("y {}", y);
                        println!("layer_activation {}", layer_activation);
                        println!("diff {}", diff);
                        */
                        diff
                    } else {
                        // If it's not the final layer, sum the errors of each neuron in the next layer
                        // that are using this activation value, weighted by the weight of the connection.
                        let next_weights = self.layers[layer + 1].weights();
                        let mut diff = 0.0;
                        /*

                        println!("---");
                        println!("activation shape {:?}", layer_activation.shape());
                        println!("activation row {} col {}", row, col);
                        println!("next_weights shape {:?}", next_weights.borrow().shape());
                        println!("next_activation shape {:?}", layer_diffs[layer + 1].shape());
                        println!("next_diffs shape {:?}", activations[layer + 1].shape());
                                                */

                        for k in 0..next_weights.borrow().cols() {
                            // TODO the 0s here are probably bad.
                            let next_weight = next_weights.as_ref().borrow().get(0, col);
                            let next_diff: f32 = layer_diffs[layer + 1].get(k, 0);
                            diff += next_weight * next_diff;
                        }
                        diff
                    };

                    let f_derivative = derived_matrix.get(row, col);
                    let delta_pi = f_derivative * target_diff;
                    layer_diffs[layer].set(row, col, delta_pi);
                }
            }

            //println!("Layer {} activation {}", layer, layer_activation);
            //println!("Layer {} Layer_diffs  {}", layer, layer_diffs[layer]);

            let layer_weights = self.layers[layer].weights();
            let weight_rows = layer_weights.borrow().rows();
            let weight_cols = layer_weights.borrow().cols();
            for row in 0..weight_rows {
                for col in 0..weight_cols {
                    // Linear is X * W^t
                    // X has tokens in rows.
                    // X cols match W cols (because of the transpose)
                    let a_pj = {
                        if layer == 0 {
                            /*
                            println!("apj thing");
                            println!("row {}, col {}", row, col);
                            println!("x {}", x);
                             */
                            x.get(row, col)
                        } else {
                            activations[layer - 1].get(row, col)
                        }
                    };
                    /*
                    println!("----");
                    println!("weights shape {:?}", layer_weights.borrow().shape());
                    println!("layer_diffs shape {:?}", layer_diffs[layer].shape());
                    println!("weights row {} col {}", row, col);
                     */

                    // TODO the indexing into layer_diffs here is probably wrong.
                    let delta_pi = layer_diffs[layer].get(row, col);
                    let delta_w_ij = learning_rate * delta_pi * a_pj;
                    weight_deltas[layer].set(row, col, delta_w_ij);
                }
            }
        }

        // Apply deltas
        for layer in 0..self.layers.len() {
            let error = (*self.layers[layer].weights())
                .borrow()
                .add(&weight_deltas[layer], &mut addition);
            match error {
                Ok(_) => {
                    *self.layers[layer].weights().as_ref().borrow_mut() = addition.clone();
                }
                _ => (),
            }
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
                        *layer_weights.borrow(),
                    );
                }
            }
        }

        previous_activation
    }
}
