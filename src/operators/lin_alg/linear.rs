use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{Add, BinaryOperator, Device, Error, MatMul, TensorWithGrad, UnaryOperator};

pub struct Linear {
    weights: TensorWithGrad,
    biases: TensorWithGrad,
    matmul: MatMul,
    add: Add,
}

impl Linear {
    pub fn new(
        device: &Device,
        weights_rows: usize,
        weights_cols: usize,
        weights_random: bool,
        bias_rows: usize,
    ) -> Result<Self, Error> {
        // Xavier Initialization, or Glorot Initialization,
        let mut rng = thread_rng();
        let right = (6.0 as f32).sqrt() / (weights_cols as f32 + weights_rows as f32).sqrt();
        let left = -right;
        let uniform = Uniform::new(left, right);

        let mut weights = Vec::new();
        weights.resize(weights_rows * weights_cols, 0.0);
        if weights_random {
            for index in 0..weights.len() {
                weights[index] = rng.sample(uniform);
            }
        }
        let weights =
            device.tensor_with_grad(weights_rows, weights_cols, weights, &[], true, true)?;

        let biases_len = bias_rows * weights_rows;
        let biases = device.tensor_with_grad(
            bias_rows,
            weights_rows,
            vec![0.0; biases_len],
            &[],
            true,
            true,
        )?;

        let transb = true;
        let op = Self {
            weights,
            biases,
            matmul: MatMul::new(device, transb),
            add: Add::new(device),
        };
        Ok(op)
    }
}

impl UnaryOperator for Linear {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let product = self.matmul.forward(input, &self.weights)?;
        let sum = self.add.forward(&product, &self.biases)?;
        Ok(sum)
    }
}
