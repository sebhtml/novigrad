use std::ops::Deref;

use crate::{
    devices::Device, gradient_instruction, inference_instruction, Error, Instruction, OpCode,
    Tensor, TensorF32, UnaryOperator,
};

#[derive(Clone)]
pub struct Reshape {
    device: Device,
    input_size: Vec<usize>,
    output_size: Vec<usize>,
}

impl Reshape {
    pub fn new(device: &Device, input_size: Vec<usize>, output_size: Vec<usize>) -> Self {
        Self {
            device: device.clone(),
            input_size,
            output_size,
        }
    }

    pub fn execute(
        output_size: &[usize],
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::copy(input, output)?;
        output.resize(output_size)
    }
}

impl UnaryOperator for Reshape {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_tensor: &TensorF32 = &input.tensor().deref().borrow();
        debug_assert_eq!(*input_tensor.size().deref().borrow_mut(), self.input_size);
        let rows = self.output_size[0];
        let cols = self.output_size[1];
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
            OpCode::Reshape(self.output_size.clone()),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [&output];
        let outputs = [input];
        output.push_instruction(gradient_instruction!(
            OpCode::ReshapeBackward(self.input_size.clone()),
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        Ok(output)
    }
}

pub struct ReshapeBackward {}

impl ReshapeBackward {
    pub fn execute(
        input_size: &[usize],
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
    ) -> Result<(), Error> {
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let input_gradient = inputs[0];
            TensorF32::copy(input_gradient, output_gradient)?;
            output_gradient.resize(input_size)?;
        }
        Ok(())
    }
}
