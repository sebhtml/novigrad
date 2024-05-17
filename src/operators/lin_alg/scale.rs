use std::{ops::Deref, rc::Rc};

use crate::{Device, Instruction, OpCode, Tensor, TensorF32, UnaryOperator, Zero};

/// Scale is not a ONNX operator. https://onnx.ai/onnx/operators/index.html ???
#[derive(Clone)]
pub struct Scale {
    device: Device,
    alpha: f32,
}

impl Scale {
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

impl UnaryOperator for Scale {
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
        output.push_instruction(Instruction::new(
            OpCode::DynOperator(Rc::new(Zero::default())),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
            crate::Category::Inference,
        ));
        output.push_instruction(Instruction::new(
            OpCode::DynOperator(Rc::new(Zero::default())),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
            crate::Category::Inference,
        ));
        output.push_instruction(Instruction::new(
            OpCode::Scale(self.alpha),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
            crate::Category::Inference,
        ));
        let inputs = [&output];
        let outputs = [input];
        output.push_instruction(Instruction::new(
            OpCode::ScaleBackward,
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
            crate::Category::Gradient,
        ));
        Ok(output)
    }
}

pub struct ScaleBackward {}

impl ScaleBackward {
    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), crate::Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::add(input, output)
    }
}
