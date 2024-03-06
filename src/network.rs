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
                    weights.resize(rows * cols, 0.0);
                    Matrix::new(*rows, *cols, weights)
                })
                .collect(),
        }
    }
    pub fn train(&mut self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {
        for i in 0..inputs.len() {
            _ = self.train_with_one_example(&inputs[i], &outputs[i]);
        }
    }

    fn train_with_one_example(&mut self, x: &Vec<f32>, y: &Vec<f32>) {
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
                            activation.set(row, col, sigmoid(activation.get(row, col)));
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

        let output = &activations[activations.len() - 1];
        let target = Matrix::new(y.len(), 1, y.clone());

        let output_vec: Vec<f32> = output.clone().into();
        let error = self.compute_error(y, &output_vec);
        println!("Error: {}", error);

        // delta rule
        println!("Applying delta rule");
        for layer in vec![self.layers.len() - 1].iter() {
            let layer = layer.to_owned();
            let layer_weights = self.layers[layer].clone();
            //self.layers.iter().enumerate().rev() {
            println!("Layer {}", layer);
            let layer_activation = &activations[layer];
            println!("Layer activation {}", layer_activation);
            for row in 0..layer_weights.rows() {
                println!("For row {}", row);
                let diff = target.get(row, 0) - layer_activation.get(row, 0);
                let activation_derivative =
                    sigmoid_derivative(matrix_products[layer - 1].get(row, 0));
                let diff_times_derivative = diff * activation_derivative;
                println!(
                    "diff {} activation_derivative {}",
                    diff, activation_derivative
                );
                for col in 0..layer_weights.cols() {
                    let delta = diff_times_derivative
                        * diff_times_derivative
                        * activations[layer - 1].get(row, col);
                    println!(
                        "Delta for layer {}, row {}, col {}, delta {}",
                        layer, row, col, delta
                    );

                    let new_weight = self.layers[layer].get(row, col) + delta;
                    self.layers[layer].set(row, col, new_weight);
                }
            }
        }
    }

    fn compute_error(&self, y: &Vec<f32>, output: &Vec<f32>) -> f32 {
        println!("compute error {} {}", y.len(), output.len());
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
        println!("predict");
        let x = x.clone();
        // TODO add constant bias
        // Add a constant for bias
        //x.push(1.0);
        let x = Matrix::new(x.len(), 1, x);
        let mut previous_activation = x;

        for layer_weights in self.layers.iter() {
            println!("Layer weights: {}", layer_weights);
            println!("Inputs: {}", previous_activation);

            let activation = layer_weights * &previous_activation;

            match activation {
                Ok(activation) => {
                    println!("Activation: {}", activation);
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
