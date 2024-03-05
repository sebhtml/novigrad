use crate::activation;
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
            self.train_with_one_example(&inputs[i], &outputs[i]);
        }
    }

    fn train_with_one_example(&self, x: &Vec<f32>, y: &Vec<f32>) {
        println!("[train_with_one_example]");
        let mut activations: Vec<Matrix> = Vec::new();
        let mut x = x.clone();
        // Add a constant for bias
        x.push(1.0);
        let x = Matrix::new(2, 1, x);

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
    }

    pub fn predict_many(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, x: &Vec<f32>) -> Vec<f32> {
        println!("predict");
        let mut x = x.clone();
        // Add a constant for bias
        x.push(1.0);
        let x = Matrix::new(2, 1, x);
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

        previous_activation.into()
    }
}
