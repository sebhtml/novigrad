use std::ops::Deref;

use crate::{
    gradient_instruction, inference_instruction, Device, Instruction, OpCode, Tensor, TensorF32,
    UnaryOperator,
};

pub struct ScalarMul {
    device: Device,
    alpha: f32,
}

impl ScalarMul {
    pub fn new(device: &Device, alpha: f32) -> Self {
        Self {
            device: device.clone(),
            alpha,
        }
    }

    pub fn execute(
        alpha: f32,
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
    ) -> Result<(), crate::Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::copy(input, output)?;
        TensorF32::scale(alpha, output)
    }
}

impl UnaryOperator for ScalarMul {
    fn forward(&self, input: &Tensor) -> Result<Tensor, crate::Error> {
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
            OpCode::ScalarMul(0.0),
            &[&outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul(0.0),
            &[&outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul(self.alpha),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [&output];
        let outputs = [input];
        output.push_instruction(gradient_instruction!(
            OpCode::ScalarMulBackward,
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        Ok(output)
    }
}

pub struct ScalarMulBackward {}

impl ScalarMulBackward {
    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), crate::Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::add(input, output)
    }
}
