use rand::Rng;

use crate::{Activation, ActivationFunction, Tensor};

pub struct LayerConfig {
    pub rows: usize,
    pub cols: usize,
    pub activation: Activation,
}

pub struct Layer {
    pub weights: Tensor,
    pub activation: Box<dyn ActivationFunction>,
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_configs: Vec<LayerConfig>) -> Self {
        Self {
            layers: layer_configs
                .into_iter()
                .map(|layer_config| {
                    let mut weights = Vec::new();
                    let rows = layer_config.rows;
                    let cols = layer_config.cols;
                    let activation = layer_config.activation;
                    weights.resize(rows * cols, 0.0);
                    for index in 0..weights.len() {
                        weights[index] = rand::thread_rng().gen_range(0.0..1.0);
                    }
                    let weights = Tensor::new(vec![rows, cols], weights);
                    Layer {
                        weights,
                        activation: activation.clone().into(),
                    }
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
            println!(
                "Example Error example {} target {:?} predicted {:?} error {}",
                i, target, predicted, example_error
            );
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
            // Use the same convention that is used in tensorflow:
            //  y= x W^T+b
            // Weights is on the right.
            // W is transposed.
            // X is not transposed.
            let matrix_product = previous_activation * &layer_weights.transpose();

            match matrix_product {
                Ok(matrix_product) => {
                    matrix_products.push(matrix_product.clone());
                    let activation = activation.activate_matrix(matrix_product);
                    activations.push(activation);
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between  X {} and W {}", previous_activation, layer_weights,);
                }
            }
        }

        // Back-propagation
        let mut weight_deltas: Vec<Tensor> =
            self.layers.iter().map(|x| x.weights.clone()).collect();
        let mut layer_diffs = Vec::new();
        layer_diffs.resize(self.layers.len(), Vec::<f32>::new());

        // TODO generalize how tensor.dimensions() is used.
        for (layer, _) in self.layers.iter().enumerate().rev() {
            let layer = layer.to_owned();
            let layer_weights = &self.layers[layer].weights;
            let activation = &self.layers[layer].activation;
            let layer_activation = &activations[layer];
            let derived_matrix = activation.derive_matrix(layer_activation.clone());
            for col in 0..layer_weights.dimensions()[0] {
                let f_derivative = derived_matrix.get(&vec![0, col]);
                let target_diff = if layer == self.layers.len() - 1 {
                    y.get(&vec![0, col]) - layer_activation.get(&vec![0, col])
                } else {
                    let next_weights = &self.layers[layer + 1].weights;
                    let mut sum = 0.0;
                    for k in 0..next_weights.dimensions()[0] {
                        let next_weight = next_weights.get(&vec![k, col]);
                        let next_diff: f32 = layer_diffs[layer + 1][k];
                        sum += next_weight * next_diff;
                    }
                    sum
                };

                let delta_pi = f_derivative * target_diff;
                layer_diffs[layer].push(delta_pi);

                for row in 0..layer_weights.dimensions()[1] {
                    let a_pj = {
                        if layer == 0 {
                            x.get(&vec![0, row])
                        } else {
                            activations[layer - 1].get(&vec![0, row])
                        }
                    };
                    let delta_w_ij = learning_rate * delta_pi * a_pj;
                    weight_deltas[layer].set(&vec![col, row], delta_w_ij);
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

    fn compute_error(&self, y: &Tensor, output: &Tensor) -> f32 {
        let mut error = 0.0;
        for i in 0..y.dimensions()[0] {
            let diff = y.get(&vec![i, 0]) - output.get(&vec![i, 0]);
            error += diff.powf(2.0);
        }
        error * 0.5
    }

    pub fn predict_many(&self, inputs: &Vec<Tensor>) -> Vec<Tensor> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, x: &Tensor) -> Tensor {
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let mut previous_activation = x.clone();

        for layer in self.layers.iter() {
            let layer_weights = &layer.weights;
            let activation = &layer.activation;
            let matrix_product = &previous_activation * &layer_weights.transpose();

            match matrix_product {
                Ok(matrix_product) => {
                    let activation = activation.activate_matrix(matrix_product);
                    previous_activation = activation;
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between  X {} and W {}", previous_activation, layer_weights,);
                }
            }
        }

        previous_activation
    }
}
