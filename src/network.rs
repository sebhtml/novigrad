use crate::activation;
use crate::matrix::ElementWiseProduct;
use crate::matrix::Error;
use crate::sigmoid;
use crate::Matrix;
pub struct Network {
    layers: Vec<Matrix>,
}

impl Network {
    pub fn new() -> Self {
        let layer_sizes = vec![(16, 2), (32, 16), (64, 32), (32, 64), (16, 32), (1, 16)];
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
    pub fn train(&self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {
        for i in 0..inputs.len() {
            _ = self.train_with_one_example(&inputs[i], &outputs[i]);
        }
    }

    fn train_with_one_example(&self, x: &Vec<f32>, y: &Vec<f32>) -> Result<Matrix, Error> {
        println!("[train_with_one_example]");
        let mut activations: Vec<Matrix> = Vec::new();
        let mut x = x.clone();
        // Add a constant for bias
        x.push(1.0);
        let x = Matrix::new(x.len(), 1, x);

        for (i, layer_weights) in self.layers.iter().enumerate() {
            let previous_activation = {
                if i == 0 {
                    &x
                } else {
                    &activations[activations.len() - 1]
                }
            };
            println!("Layer weights: {}", layer_weights);
            println!("Inputs: {}", previous_activation);

            let activation = layer_weights * previous_activation;

            match activation {
                Ok(activation) => {
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
        let previous_output = &activations[activations.len() - 2];
        let target = Matrix::new(y.len(), 1, y.clone());

        let output_vec: Vec<f32> = output.clone().into();
        let error = self.compute_error(y, &output_vec);
        println!("Error: {}", error);

        // delta rule
        // TODO fix this equation of the delta rule
        let d_error_d_weights = &(&(&-(&target - output)? * output)? * (/*1.0 -*/output))?
            .element_wise_product(previous_output)?;
        println!("d_error_d_weights for last layer: {}", d_error_d_weights);
        Ok(d_error_d_weights.to_owned())
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
        println!("predict");
        let mut x = x.clone();
        // Add a constant for bias
        x.push(1.0);
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
