use rand::Rng;

use crate::{Activation, ActivationFunction, Matrix};

pub struct LayerConfig {
    pub size: usize, // neurons
    pub activation: Activation,
}

pub struct Layer {
    pub weights: Matrix,
    pub activation: Box<dyn ActivationFunction>,
}

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<LayerConfig>) -> Self {
        let mut layer_configs = Vec::new();
        for index in 1..layers.len() {
            layer_configs.push((
                layers[index].size,
                layers[index - 1].size,
                layers[index].activation.clone(),
            ));
        }
        Self {
            layers: layer_configs
                .iter()
                .map(|(rows, cols, activation)| {
                    let mut weights = Vec::new();
                    weights.resize(rows * cols, 0.0);
                    for index in 0..weights.len() {
                        weights[index] = rand::thread_rng().gen_range(0.0..1.0);
                    }
                    let weights = Matrix::new(*rows, *cols, weights);
                    Layer {
                        weights,
                        activation: activation.clone().into(),
                    }
                })
                .collect(),
        }
    }

    pub fn train(&mut self, inputs: &Vec<Matrix>, outputs: &Vec<Matrix>) {
        for i in 0..inputs.len() {
            self.train_back_propagation(i, &inputs[i], &outputs[i]);
        }
    }

    pub fn total_error(&self, inputs: &Vec<Matrix>, outputs: &Vec<Matrix>) -> f32 {
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
    fn train_back_propagation(&mut self, _example: usize, x: &Matrix, y: &Matrix) {
        let learning_rate = 0.5;
        let mut matrix_products: Vec<Matrix> = Vec::new();
        let mut activations: Vec<Matrix> = Vec::new();
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let previous_activation = {
                if layer_index == 0 {
                    &x
                } else {
                    &activations[activations.len() - 1]
                }
            };

            let layer_weights = &layer.weights;
            let activation = &layer.activation;
            let matrix_product = layer_weights * previous_activation;

            match matrix_product {
                Ok(matrix_product) => {
                    matrix_products.push(matrix_product.clone());
                    let activation = activation.activate_matrix(matrix_product);
                    activations.push(activation);
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between  W {} and A {}", layer_weights, previous_activation,);
                }
            }
        }

        // Back-propagation
        let mut weight_deltas: Vec<Matrix> =
            self.layers.iter().map(|x| x.weights.clone()).collect();
        let mut layer_diffs = Vec::new();
        layer_diffs.resize(self.layers.len(), Vec::<f32>::new());

        for (layer, _) in self.layers.iter().enumerate().rev() {
            let layer = layer.to_owned();
            let layer_weights = &self.layers[layer].weights;
            let activation = &self.layers[layer].activation;
            let layer_activation = &activations[layer];
            let derived_matrix = activation.derive_matrix(layer_activation.clone());
            for row in 0..layer_weights.rows() {
                let f_derivative = derived_matrix.get(row, 0);
                let target_diff = if layer == self.layers.len() - 1 {
                    y.get(row, 0) - layer_activation.get(row, 0)
                } else {
                    let next_weights = &self.layers[layer + 1].weights;
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
            match &self.layers[layer].weights + &weight_deltas[layer] {
                Ok(matrix) => {
                    self.layers[layer].weights = matrix;
                }
                _ => (),
            }
        }
    }

    fn compute_error(&self, y: &Matrix, output: &Matrix) -> f32 {
        let mut error = 0.0;
        for i in 0..y.rows() {
            let diff = y.get(i, 0) - output.get(i, 0);
            error += diff.powf(2.0);
        }
        error * 0.5
    }

    pub fn predict_many(&self, inputs: &Vec<Matrix>) -> Vec<Matrix> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, x: &Matrix) -> Matrix {
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let mut previous_activation = x.clone();

        for layer in self.layers.iter() {
            let layer_weights = &layer.weights;
            let activation = &layer.activation;
            let matrix_product = layer_weights * &previous_activation;

            match matrix_product {
                Ok(matrix_product) => {
                    let activation = activation.activate_matrix(matrix_product);
                    previous_activation = activation;
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between  W {} and A {}", layer_weights, previous_activation,);
                }
            }
        }

        previous_activation
    }
}
