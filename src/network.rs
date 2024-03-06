use crate::{activation::sigmoid, sigmoid_derivative, Matrix};
pub struct Network {
    layers: Vec<Matrix>,
}

impl Network {
    pub fn new() -> Self {
        let layer_sizes = vec![(16, 4), (32, 16), (16, 32), (1, 16)];
        Self {
            layers: layer_sizes
                .iter()
                .map(|(rows, cols)| -> Matrix {
                    let mut weights = Vec::new();
                    weights.resize(rows * cols, 0.5);
                    Matrix::new(*rows, *cols, weights)
                })
                .collect(),
        }
    }
    pub fn train(&mut self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {
        for i in 0..inputs.len() {
            self.train_with_one_example(i, &inputs[i], &outputs[i]);
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

    fn train_with_one_example(&mut self, _example: usize, x: &Vec<f32>, y: &Vec<f32>) {
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

        // delta rule
        let mut deltas = self.layers.clone();
        let mut layer_diffs = Vec::new();
        layer_diffs.resize(self.layers.len(), Vec::<f32>::new());
        println!("Applying delta rule");
        for (layer, _) in self.layers.iter().enumerate().rev() {
            let layer = layer.to_owned();
            let layer_weights = self.layers[layer].clone();
            //self.layers.iter().enumerate().rev() {
            println!("Layer {}", layer);
            let layer_activation = &activations[layer];
            println!("Layer activation {}", layer_activation);
            println!("layer weights {}", layer_weights);
            for row in 0..layer_weights.rows() {
                println!("For row {}", row);
                let diff = if layer == self.layers.len() - 1 {
                    y[row] - layer_activation.get(row, 0)
                } else {
                    let mut diff = 0.0;
                    let next_weights = &self.layers[layer + 1];
                    let next_diffs = &layer_diffs[layer + 1];
                    println!("next weights {}, next diffs {:?}", next_weights, next_diffs);
                    for index in 0..next_diffs.len() {
                        // For that diff in the next layer neuron,
                        // take the contribution of the current neuron.
                        let mut total_weight = 0.0;

                        for next_col in 0..next_weights.cols() {
                            total_weight += next_weights.get(index, next_col);
                        }
                        println!("Current layer: {}", layer);
                        println!("next_diffs len {}", next_diffs.len());
                        println!("layer weight shape {:?}", layer_weights.shape());
                        println!("next_weights shape {:?}", next_weights.shape());
                        let my_weight = next_weights.get(index, row);
                        let contribution = my_weight / total_weight * layer_diffs[layer + 1][index];
                        if contribution.is_finite() {
                            diff += contribution;
                        }
                    }
                    diff
                };
                layer_diffs[layer].push(diff);
                println!("Pushed diff {} for layer {}", diff, layer);

                for col in 0..layer_weights.cols() {
                    let activation_derivative = if layer == 0 {
                        0.0 // TODO
                    } else {
                        sigmoid_derivative(matrix_products[layer - 1].get(col, 0))
                    };

                    let input_i = if layer != 0 {
                        let previous_activation = activations[layer - 1].clone();
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

                    deltas[layer].set(row, col, delta);
                }
            }
        }

        for layer in 0..self.layers.len() {
            match &self.layers[layer] + &deltas[layer] {
                Ok(matrix) => {
                    self.layers[layer] = matrix;
                    println!("Updated matrix at layer {} with deltas", layer);
                    println!("{}", deltas[layer]);
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
