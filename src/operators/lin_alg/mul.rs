use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, OperatorTrait, Tensor, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Mul.html
#[derive(Clone)]
pub struct Mul {
    device: Device,
}

impl Mul {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl OperatorTrait for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
        debug_assert_eq!(input_0.size(), input_1.size());
        let rows = input_0.rows();
        let cols = input_0.cols();
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
        let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        TensorF32::mul(input_0, input_1, output)
    }

    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();

        if inputs[1].requires_grad() {
            let input_1_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
            let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
            TensorF32::mul(input_0, output_gradient, input_1_gradient)?;
        }

        if inputs[0].requires_grad() {
            let input_0_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
            TensorF32::mul(input_1, output_gradient, input_0_gradient)?;
        }

        Ok(())
    }
}
