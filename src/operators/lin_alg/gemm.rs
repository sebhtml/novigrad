use std::{ops::Deref, rc::Rc};

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{devices::Device, Error, Identity, OperatorTrait, Tensor, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Gemm.html
#[derive(Clone)]
pub struct Gemm {
    device: Device,
    weights: Tensor,
    biases: Tensor,
}

impl Gemm {
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

        Gemm {
            device: device.clone(),
            weights,
            biases,
        }
    }
}

impl OperatorTrait for Gemm {
    fn name(&self) -> &str {
        "Linear"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 1);
        let biases: &TensorF32 = &self.biases.tensor().deref().borrow();
        let rows = biases.rows();
        let cols = biases.cols();
        let len = rows * cols;
        let output = self.device.tensor(
            Rc::new(self.clone()),
            inputs,
            rows,
            cols,
            vec![0.0; len],
            true,
            false,
        );

        Ok(output)
    }

    fn forward_realize(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let biases: &TensorF32 = &self.biases.tensor().deref().borrow();
        // Use the same convention that is used in tensorflow:
        // Y = X @ W^T + B
        // Weights is on the right.
        // X is not transposed.
        // W is transposed.

        // use GEMM to do C = A * W^T + C  with weights and biases all together.
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        let weights: &TensorF32 = &self.weights.tensor().deref().borrow();
        let a = input;
        let b = weights;
        let c = output;
        TensorF32::copy(biases, c)?;
        TensorF32::gemm(false, true, 1.0, a, b, 1.0, c, false)
    }

    fn backward(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();

        if self.weights.requires_grad() {
            let weights_gradient: &mut TensorF32 =
                &mut self.weights.gradient().deref().borrow_mut();
            let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
            let a: &TensorF32 = input;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = weights_gradient;
            TensorF32::gemm(true, false, 1.0, a, b, 1.0, c, true)?;
        }

        if self.biases.requires_grad() {
            let biases_gradient: &mut TensorF32 = &mut self.biases.gradient().deref().borrow_mut();
            TensorF32::add(output_gradient, biases_gradient)?;
        }

        if inputs[0].requires_grad() {
            let input_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let weights: &TensorF32 = &self.weights.tensor().deref().borrow();
            let a: &TensorF32 = weights;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = input_gradient;
            TensorF32::gemm(true, true, 1.0, a, b, 1.0, c, true)?;
        }

        Ok(())
    }
}
