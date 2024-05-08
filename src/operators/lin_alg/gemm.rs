use std::{ops::Deref, rc::Rc};

use crate::{devices::Device, Error, OperatorTrait, Tensor, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Gemm.html
#[derive(Clone)]
pub struct Gemm {
    device: Device,
}

impl Gemm {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl OperatorTrait for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 3);
        let biases = &inputs[2];
        let biases: &TensorF32 = &biases.tensor().deref().borrow();
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
        // Use the same convention that is used in tensorflow:
        // Y = X @ W^T + B
        // Weights is on the right.
        // X is not transposed.
        // W is transposed.

        // use GEMM to do C = A * W^T + C  with weights and biases all together.
        debug_assert_eq!(inputs.len(), 3);
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let weights: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let biases: &TensorF32 = &inputs[2].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        let a = input;
        let b = weights;
        let c = output;
        TensorF32::copy(biases, c)?;
        TensorF32::gemm(false, true, 1.0, a, b, 1.0, c, false)
    }

    fn backward(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 3);
        let input = &inputs[0];
        let weights = &inputs[1];
        let biases = &inputs[2];
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();

        if weights.requires_grad() {
            let weights_gradient: &mut TensorF32 = &mut weights.gradient().deref().borrow_mut();
            let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
            let a: &TensorF32 = input;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = weights_gradient;
            TensorF32::gemm(true, false, 1.0, a, b, 1.0, c, true)?;
        }

        if biases.requires_grad() {
            let biases_gradient: &mut TensorF32 = &mut biases.gradient().deref().borrow_mut();
            TensorF32::add(output_gradient, biases_gradient)?;
        }

        if input.requires_grad() {
            let input_gradient: &mut TensorF32 = &mut input.gradient().deref().borrow_mut();
            let weights: &TensorF32 = &weights.tensor().deref().borrow();
            let a: &TensorF32 = weights;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = input_gradient;
            TensorF32::gemm(true, true, 1.0, a, b, 1.0, c, true)?;
        }

        Ok(())
    }
}
