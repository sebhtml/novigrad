use std::rc::Rc;

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{Device, Error, Gemm, Identity, OperatorTrait, Tensor};

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
        bias_rows: usize,
    ) -> Self {
        // Xavier Initialization, or Glorot Initialization,
        let mut rng = thread_rng();
        let right = (6.0 as f32).sqrt() / (weights_cols as f32 + weights_rows as f32).sqrt();
        let left = -right;
        let uniform = Uniform::new(left, right);

        let mut weights = Vec::new();
        weights.resize(weights_rows * weights_cols, 0.0);
        for index in 0..weights.len() {
            weights[index] = rng.sample(uniform);
        }
        let weights = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            weights_rows,
            weights_cols,
            weights,
            true,
            true,
        );

        let biases_len = bias_rows * weights_rows;
        let biases = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            bias_rows,
            weights_rows,
            vec![0.0; biases_len],
            true,
            true,
        );

        let gemm = Gemm::new(device);
        Self {
            weights,
            biases,
            gemm,
        }
    }
}

impl OperatorTrait for Linear {
    fn name(&self) -> &str {
        "Linear"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 1);
        let inputs = &[inputs[0].clone(), self.weights.clone(), self.biases.clone()];
        self.gemm.forward(inputs)
    }

    fn forward_realize(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        panic!()
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        panic!()
    }
}
