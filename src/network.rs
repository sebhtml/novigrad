use rand::Rng;

use crate::{activation::sigmoid, sigmoid_derivative, Matrix};
pub struct Network {
    layers: Vec<Matrix>,
}

impl Network {
    pub fn new() -> Self {
        let layer_sizes = vec![(16, 4), (1, 16)];
        //  let layer_sizes = vec![(16, 4), (32, 16), (16, 32), (1, 16)];
        //let layer_sizes = vec![(1, 4)];
        Self {
            layers: layer_sizes
                .iter()
                .map(|(rows, cols)| -> Matrix {
                    let mut weights = Vec::new();
                    weights.resize(rows * cols, 0.0);
                    for index in 0..weights.len() {
                        weights[index] = rand::thread_rng().gen_range(0.0..1.0);
                    }
                    Matrix::new(*rows, *cols, weights)
                })
                .collect(),
        }
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {
        for i in 0..inputs.len() {
            self.train_back_propagation(i, &inputs[i], &outputs[i]);
        }
    }

    pub fn total_error(&self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) -> f32 {
        let mut total_error = 0.0;
        for i in 0..inputs.len() {
            let predicted = self.predict(&inputs[i]);
            let target = &outputs[i];
            let example_error = self.compute_error(target, &predicted);
            println!(
                "Example Error example {} target {:?} predicted {:?} error {}",
                i, target, predicted, example_error
            );
            total_error += example_error;
        }

        total_error
    }

    fn train_back_propagation(&mut self, _example: usize, x: &Vec<f32>, y: &Vec<f32>) {
        let learning_rate = 0.5;
        println!("[train_with_one_example]");
        let mut matrix_products: Vec<Matrix> = Vec::new();
        let mut activations: Vec<Matrix> = Vec::new();
        let x = x.clone();
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let x = Matrix::new(x.len(), 1, x);

        for (layer, layer_weights) in self.layers.iter().enumerate() {
            let previous_activation = {
                if layer == 0 {
                    &x
                } else {
                    &activations[activations.len() - 1]
                }
            };
            println!("Layer {} weights: {}", layer, layer_weights);
            println!("Inputs: {}", previous_activation);

            let matrix_product = layer_weights * previous_activation;

            match matrix_product {
                Ok(matrix_product) => {
                    matrix_products.push(matrix_product.clone());
                    let mut activation = matrix_product.clone();
                    for row in 0..activation.rows() {
                        for col in 0..activation.cols() {
                            activation.set(row, col, sigmoid(matrix_product.get(row, col)));
                        }
                    }
                    println!("matrix_product: {}", matrix_product);
                    println!("Activation: {}", activation);
                    activations.push(activation);
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between  W {} and A {}", layer_weights, previous_activation,);
                }
            }
        }

        // Back-propagation
        // delta rule
        let mut weight_deltas = self.layers.clone();
        let mut layer_diffs = Vec::new();
        layer_diffs.resize(self.layers.len(), Vec::<f32>::new());
        println!("Applying delta rule");
        for (layer, _) in self.layers.iter().enumerate().rev() {
            /*
             product =  | a b c d | * | i |  = | n |
                        | e f g h |   | k |    | o |
                                      | l |
                                      | m |

            activation = | sigmoid(n) |
                         | sigmoid(o) |

                       */
            let layer = layer.to_owned();
            let layer_weights = &self.layers[layer];

            println!("Layer {}", layer);
            let layer_activation = &activations[layer];
            let matrix_product = &matrix_products[layer];
            println!("Layer activation {}", layer_activation);
            println!("layer weights {}", layer_weights);
            for row in 0..layer_weights.rows() {
                println!("For row {}", row);

                let diff = if layer == self.layers.len() - 1 {
                    let diff = y[row] - layer_activation.get(row, 0);
                    diff
                } else {
                    // The next layer has N neurons, each with a back-propagated error.
                    // Each neuron in the current layer contributes to the error of
                    // each neuron in the next layer.
                    // In the current layer weights, if we change W_ij, how much does it affect
                    // the errors of the next layer ?
                    let mut sum_of_diffs = 0.0;
                    let next_weights = &self.layers[layer + 1];
                    let next_diffs = &layer_diffs[layer + 1];
                    println!("next_weights {}", next_weights);
                    println!("next_diffs {:?}", next_diffs);
                    for (diff_index, diff) in next_diffs.iter().enumerate() {
                        let mut sum_of_weights = 0.0;

                        for weight_index in 0..next_weights.cols() {
                            sum_of_weights += layer_activation.get(diff_index, 0)
                                * next_weights.get(diff_index, weight_index);
                        }

                        println!(
                            "layer {} row {} sum_of_weights {}",
                            layer, row, sum_of_weights
                        );
                        let my_weight =
                            layer_activation.get(diff_index, 0) * next_weights.get(diff_index, row);
                        println!("my_weight {}", my_weight);
                        let contribution = my_weight / sum_of_weights * diff;
                        if contribution.is_finite() {
                            sum_of_diffs += contribution;
                        }
                    }
                    sum_of_diffs
                };
                layer_diffs[layer].push(diff);

                for col in 0..layer_weights.cols() {
                    println!("Pushed diff {} for layer {}", diff, layer);

                    let activation_derivative = if layer == 0 {
                        1.0 // TODO is this the correct thing ?
                    } else {
                        println!("previous matrix product {}", matrix_products[layer - 1]);

                        sigmoid_derivative(matrix_products[layer - 1].get(col, 0))
                    };

                    let input_i = if layer != 0 {
                        let previous_activation = &activations[layer - 1];
                        println!("Previous activation {}", previous_activation);
                        previous_activation.get(col, 0)
                    } else {
                        x.get(col, 0)
                    };
                    let delta = learning_rate * diff * activation_derivative * input_i;
                    println!(
                        "Delta for layer {}, row {}, col {}, diff {}, activation_derivative {}, input_i {}, delta {}",
                        layer, row, col, diff, activation_derivative, input_i, delta,
                    );

                    weight_deltas[layer].set(row, col, delta);
                }
            }
        }

        for (layer, diffs) in layer_diffs.iter().enumerate() {
            println!("DEBUG Layer {} diffs {:?}", layer, diffs);
        }
        for layer in 0..self.layers.len() {
            match &self.layers[layer] + &weight_deltas[layer] {
                Ok(matrix) => {
                    // TODO update all layers
                    //if layer == self.layers.len() - 1 {
                    self.layers[layer] = matrix;
                    //}
                }
                _ => (),
            }
        }
    }

    fn compute_error(&self, y: &Vec<f32>, output: &Vec<f32>) -> f32 {
        let mut error = 0.0;
        for i in 0..y.len() {
            let diff = y[i] - output[i];
            error += diff.powf(2.0);
        }
        error * 0.5
    }

    pub fn predict_many(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, x: &Vec<f32>) -> Vec<f32> {
        let x = x.clone();
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let x = Matrix::new(x.len(), 1, x);
        let mut previous_activation = x;

        for layer_weights in self.layers.iter() {
            let matrix_product = layer_weights * &previous_activation;

            match matrix_product {
                Ok(matrix_product) => {
                    let mut activation = matrix_product.clone();
                    for row in 0..activation.rows() {
                        for col in 0..activation.cols() {
                            activation.set(row, col, sigmoid(matrix_product.get(row, col)));
                        }
                    }
                    previous_activation = activation;
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between  W {} and A {}", layer_weights, previous_activation,);
                }
            }
        }

        let output: Vec<f32> = previous_activation.into();
        output
    }
}
