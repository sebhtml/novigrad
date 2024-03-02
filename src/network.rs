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
    pub fn train(&self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {}

    pub fn back_propagation(&self, input: &Vec<f32>, output: &Vec<f32>, target: &Vec<f32>) {}

    pub fn predict_many(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        println!("predict");
        let mut activations: Vec<Matrix> = Vec::new();
        let mut input = input.clone();
        // Add a constant for bias
        input.push(1.0);
        let activation = Matrix::new(2, 1, input);
        activations.push(activation);
        for layer_weights in self.layers.iter() {
            println!("Layer weights: {}", layer_weights);
            println!("Inputs: {}", &activations[activations.len() - 1]);

            let activation = layer_weights * &activations[activations.len() - 1];

            match activation {
                Ok(activation) => {
                    println!("Activations: {}", activation);
                    activations.push(activation);
                }
                _ => {
                    println!("Incompatible shapes in matrix multiplication");
                    println!(
                        "Between  W {} and A {}",
                        layer_weights,
                        &activations[activations.len() - 1]
                    );
                }
            }
        }

        activations[activations.len() - 1].clone().into()
    }
}
