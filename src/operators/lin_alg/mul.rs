use std::{ops::Deref, rc::Rc};

use crate::{BinaryOperator, Device, Error, Instruction, Operator, Tensor, TensorF32};

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

impl BinaryOperator for Mul {
    fn forward(&self, input_0: &Tensor, input_1: &Tensor) -> Result<Tensor, Error> {
        let input_0_t: &TensorF32 = &input_0.tensor().deref().borrow();
        let input_1_t: &TensorF32 = &input_1.tensor().deref().borrow();
        debug_assert_eq!(input_0_t.size(), input_1_t.size());
        let rows = input_0_t.rows();
        let cols = input_0_t.cols();
        let len = rows * cols;
        let output = self.device.tensor(rows, cols, vec![0.0; len], true, false);
        let inputs = &[input_0, input_1];
        let outputs = &[&output];
        output.push_forward_instruction(Rc::new(self.clone()), inputs, outputs);
        output.push_backward_instruction(Rc::new(MulBackward::default()), outputs, inputs);
        Ok(output)
    }
}

impl Operator for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input_0 = &inputs[0].tensor().deref().borrow();
        let input_1 = &inputs[1].tensor().deref().borrow();
        let output = &outputs[0].tensor().deref().borrow();
        TensorF32::mul(input_0, input_1, output)
    }
}

pub struct MulBackward {}

impl Default for MulBackward {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for MulBackward {
    fn name(&self) -> &str {
        "MulBackward"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        debug_assert_eq!(outputs.len(), 2);
        let input_gradient: &TensorF32 = &inputs[0].gradient().deref().borrow();

        if outputs[1].requires_grad() {
            let output_1_gradient: &mut TensorF32 = &mut outputs[1].gradient().deref().borrow_mut();
            let output_0: &TensorF32 = &outputs[0].tensor().deref().borrow();
            TensorF32::mul(output_0, input_gradient, output_1_gradient)?;
        }

        if outputs[0].requires_grad() {
            let output_0_gradient: &mut TensorF32 = &mut outputs[0].gradient().deref().borrow_mut();
            let output: &TensorF32 = &outputs[1].tensor().deref().borrow();
            TensorF32::mul(output, input_gradient, output_0_gradient)?;
        }

        Ok(())
    }
}
