use rand::Rng;

use crate::{ActivationFunction, Matrix, Sigmoid, Softmax};
pub struct Network {
    layers: Vec<Matrix>,
    activation: Box<dyn ActivationFunction>,
}

impl Network {
    pub fn new() -> Self {
        //let layer_sizes = vec![4, 1];
        //let layer_sizes = vec![4, 16, 1];
        //let layer_sizes = vec![4, 8, 8, 1];
        let layer_sizes = vec![4, 16, 16, 2];

        let mut layer_size_pairs = Vec::new();
        for index in 1..layer_sizes.len() {
            layer_size_pairs.push((layer_sizes[index], layer_sizes[index - 1]));
        }
        Self {
            layers: layer_size_pairs
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
            activation: Box::new(Softmax::default()),
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

    // https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf
    fn train_back_propagation(&mut self, _example: usize, x: &Vec<f32>, y: &Vec<f32>) {
        let learning_rate = 0.5;
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

            let matrix_product = layer_weights * previous_activation;

            match matrix_product {
                Ok(matrix_product) => {
                    matrix_products.push(matrix_product.clone());
                    let activation = self.activation.activate_matrix(matrix_product);
                    activations.push(activation);
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between  W {} and A {}", layer_weights, previous_activation,);
                }
            }
        }

        // Back-propagation
        let mut weight_deltas = self.layers.clone();
        let mut layer_diffs = Vec::new();
        layer_diffs.resize(self.layers.len(), Vec::<f32>::new());

        for (layer, _) in self.layers.iter().enumerate().rev() {
            let layer = layer.to_owned();
            let layer_weights = &self.layers[layer];

            let layer_matrix_product = &matrix_products[layer];
            let layer_activation = &activations[layer];

            let derived_matrix = self.activation.derive_matrix(layer_matrix_product.clone());
            for row in 0..layer_weights.rows() {
                let f_derivative = derived_matrix.get(row, 0);
                let target_diff = if layer == self.layers.len() - 1 {
                    y[row] - layer_activation.get(row, 0)
                } else {
                    let next_weights = &self.layers[layer + 1];
                    let mut sum = 0.0;
                    for k in 0..next_weights.rows() {
                        let next_weight = next_weights.get(k, row);
                        let next_diff: f32 = layer_diffs[layer + 1][k];
                        sum += next_weight * next_diff;
                    }
                    sum
                };

                let delta_pi = f_derivative * target_diff;
                layer_diffs[layer].push(delta_pi);

                for col in 0..layer_weights.cols() {
                    let a_pj = {
                        if layer == 0 {
                            x.get(col, 0)
                        } else {
                            activations[layer - 1].get(col, 0)
                        }
                    };
                    let delta_w_ij = learning_rate * delta_pi * a_pj;
                    weight_deltas[layer].set(row, col, delta_w_ij);
                }
            }
        }

        for layer in 0..self.layers.len() {
            match &self.layers[layer] + &weight_deltas[layer] {
                Ok(matrix) => {
                    self.layers[layer] = matrix;
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
                    let activation = self.activation.activate_matrix(matrix_product);
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
