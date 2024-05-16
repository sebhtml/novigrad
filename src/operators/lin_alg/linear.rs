use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{Device, Error, Gemm, Tensor, TernaryOperator, UnaryOperator};

/// Linear is not a ONNX operator.
/// https://onnx.ai/onnx/operators/index.html ???
#[derive(Clone)]
pub struct Linear {
    weights: Tensor,
    biases: Tensor,
    gemm: Gemm,
}

impl Linear {
    pub fn new(
        device: &Device,
        weights_rows: usize,
        weights_cols: usize,
        weights_random: bool,
        bias_rows: usize,
    ) -> Self {
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
        let weights = device.tensor(weights_rows, weights_cols, weights, &[], true, true);

        let biases_len = bias_rows * weights_rows;
        let biases = device.tensor(
            bias_rows,
            weights_rows,
            vec![0.0; biases_len],
            &[],
            true,
            true,
        );

        let gemm = Gemm::new(device, true);
        Self {
            weights,
            biases,
            gemm,
        }
    }
}

impl UnaryOperator for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        self.gemm.forward(input, &self.weights, &self.biases)
    }
}
