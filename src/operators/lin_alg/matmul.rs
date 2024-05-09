use std::{ops::Deref, rc::Rc};

use crate::{devices::Device, Error, OperatorTrait, Tensor, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__MatMul.html
#[derive(Clone)]
pub struct MatMul {
    device: Device,
    transb: bool,
}

impl MatMul {
    pub fn new(device: &Device, transb: bool) -> Self {
        MatMul {
            device: device.clone(),
            transb,
        }
    }
}

impl OperatorTrait for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let rows = input_0.rows();
        let transb = self.transb;
        let cols = if transb {
            input_1.rows()
        } else {
            input_1.cols()
        };
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

    fn forward_realize(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        let a = input_0;
        let b = input_1;
        let c = output;
        let transb = self.transb;
        TensorF32::matmul(false, transb, a, b, c, false)
    }

    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();

        if inputs[1].requires_grad() {
            let input_1_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
            let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
            let a: &TensorF32 = input_0;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = input_1_gradient;
            let transb = self.transb;
            TensorF32::gemm(true, false, 1.0, a, b, 1.0, c, transb)?;
        }

        if inputs[0].requires_grad() {
            let input_0_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
            let a: &TensorF32 = input_1;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = input_0_gradient;
            let transb = self.transb;
            if transb {
                TensorF32::gemm(true, true, 1.0, a, b, 1.0, c, true)?;
            } else {
                TensorF32::gemm(true, false, 1.0, a, b, 1.0, c, true)?;
            }
        }

        Ok(())
    }
}
