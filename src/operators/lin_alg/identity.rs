use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, Instruction, Operator, Tensor, TensorF32, UnaryOperator};

/// https://onnx.ai/onnx/operators/onnx__Identity.html
#[derive(Clone)]
pub struct Identity {
    device: Device,
}

impl Identity {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl UnaryOperator for Identity {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_t: &TensorF32 = &input.tensor().deref().borrow();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output = self.device.tensor(rows, cols, vec![0.0; len], true, false);
        let inputs = &[input];
        let outputs = &[&output];
        output.push_forward_instruction(Rc::new(self.clone()), inputs, outputs);
        output.push_backward_instruction(Rc::new(IdentityBackward::default()), outputs, inputs);
        Ok(output)
    }
}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0].tensor().deref().borrow();
        let output = outputs[0].tensor().deref().borrow();
        TensorF32::copy(&input, &output)
    }
}

pub struct IdentityBackward {}

impl Default for IdentityBackward {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for IdentityBackward {
    fn name(&self) -> &str {
        "IdentityBackward"
    }

    fn forward(&self, inputs: &[&Tensor], outputs_: &[&Tensor]) -> Result<(), Error> {
        if outputs_[0].requires_grad() {
            let output_gradient: &mut TensorF32 = &mut outputs_[0].gradient().deref().borrow_mut();
            let input_gradient: &TensorF32 = &inputs[0].gradient().deref().borrow();
            TensorF32::copy(input_gradient, output_gradient)?;
        }

        Ok(())
    }
}
