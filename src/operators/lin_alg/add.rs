use std::ops::Deref;

use crate::{
    gradient_instruction, inference_instruction, BinaryOperator, Device, Error, Instruction,
    OpCode, Tensor, TensorF32,
};

pub struct Add {
    device: Device,
}

impl Add {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.to_owned(),
        }
    }

    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        TensorF32::copy(input_0, output)?;
        TensorF32::add(input_1, output)
    }
}

impl BinaryOperator for Add {
    fn forward(&self, input_1: &Tensor, input_2: &Tensor) -> Result<Tensor, Error> {
        let input_0_t: &TensorF32 = &input_1.tensor().deref().borrow();
        let input_1_t: &TensorF32 = &input_1.tensor().deref().borrow();
        debug_assert_eq!(input_0_t.size(), input_1_t.size());
        let rows = input_0_t.rows();
        let cols = input_0_t.cols();
        let len = rows * cols;
        let output =
            self.device
                .tensor(rows, cols, vec![0.0; len], &[input_1, input_2], true, false);
        let inputs = [input_1, input_2];
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
            OpCode::Add,
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [&output];
        let outputs = [input_1, input_2];
        output.push_instruction(gradient_instruction!(
            OpCode::AddBackward,
            &[&inputs[0].gradient().deref().borrow()],
            &[
                &outputs[0].gradient().deref().borrow(),
                &outputs[1].gradient().deref().borrow(),
            ],
        ));
        Ok(output)
    }
}

pub struct AddBackward {}

impl AddBackward {
    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        debug_assert_eq!(outputs.len(), 2);
        let input_gradient = inputs[0];

        if outputs[1].requires_grad() {
            let output_1_gradient = outputs[1];
            TensorF32::add(input_gradient, output_1_gradient)?;
        }

        if outputs[0].requires_grad() {
            let output_0_gradient = outputs[0];
            TensorF32::add(input_gradient, output_0_gradient)?;
        }

        Ok(())
    }
}
