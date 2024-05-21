use std::ops::Deref;

use crate::{
    devices::Device, gradient_instruction, inference_instruction, Error, GenericTensor,
    Instruction, OpCode, Tensor, UnaryOperator,
};

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
        inputs: &[&GenericTensor],
        outputs: &[&GenericTensor],
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        GenericTensor::copy(input, output)?;
        output.resize(output_size)
    }
}

impl UnaryOperator for Reshape {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_tensor: &GenericTensor = &input.tensor().deref().borrow();
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
            OpCode::Reshape(self.output_size.clone()),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [&output];
        let outputs = [input];

        if outputs[0].gradient().deref().borrow().requires_grad() {
            output.push_instruction(gradient_instruction!(
                OpCode::Reshape(self.input_size.clone()),
                &[&inputs[0].gradient().deref().borrow()],
                &[&outputs[0].gradient().deref().borrow()],
            ));
        }

        Ok(output)
    }
}
