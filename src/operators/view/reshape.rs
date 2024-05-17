use std::{ops::Deref, rc::Rc};

use crate::{
    devices::Device, Error, Instruction, Operator, Tensor, TensorF32, UnaryOperator, Zero,
};

/// https://onnx.ai/onnx/operators/onnx__Reshape.html
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
        output.push_instruction(Instruction::new(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        ));
        output.push_instruction(Instruction::new(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
            false,
        ));
        output.push_instruction(Instruction::new(
            Rc::new(self.clone()),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        ));
        let inputs = [&output];
        let outputs = [input];
        output.push_instruction(Instruction::new(
            Rc::new(ReshapeBackward::new(self.input_size.clone())),
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
            true,
        ));
        Ok(output)
    }
}

impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::copy(input, output)?;
        output.resize(&self.output_size)
    }
}

pub struct ReshapeBackward {
    input_size: Vec<usize>,
}

impl ReshapeBackward {
    pub fn new(input_size: Vec<usize>) -> Self {
        Self { input_size }
    }
}

impl Operator for ReshapeBackward {
    fn name(&self) -> &str {
        "ReshapeBackward"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let input_gradient = inputs[0];
            TensorF32::copy(input_gradient, output_gradient)?;
            output_gradient.resize(&self.input_size)?;
        }
        Ok(())
    }
}
