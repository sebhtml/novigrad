use std::ops::Deref;

use crate::{
    gradient_instruction, inference_instruction, Device, Error, Instruction, OpCode, Tensor,
    TensorF32, UnaryOperator,
};

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

    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::copy(&input, &output)
    }
}

impl UnaryOperator for Identity {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_t: &TensorF32 = &input.tensor().deref().borrow();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output = self
            .device
            .tensor(rows, cols, vec![0.0; len], &[input], true, false);
        let inputs = [input];
        let outputs = [&output];
        output.push_instruction(inference_instruction!(
            OpCode::Scale(0.0),
            &[&outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::Scale(0.0),
            &[&outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::Identity,
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [&output];
        let outputs = [input];
        output.push_instruction(gradient_instruction!(
            OpCode::IdentityBackward,
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        Ok(output)
    }
}

pub struct IdentityBackward {}

impl IdentityBackward {
    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let input_gradient = inputs[0];
            TensorF32::add(input_gradient, output_gradient)?;
        }

        Ok(())
    }
}
