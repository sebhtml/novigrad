use std::ops::Deref;

use crate::{
    gradient_instruction, inference_instruction, Device, GenericTensor, OpCode, Tensor,
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
        inputs: &[&GenericTensor],
        outputs: &[&GenericTensor],
    ) -> Result<(), crate::Error> {
        let input = inputs[0];
        let output = outputs[0];
        GenericTensor::copy(input, output)?;
        GenericTensor::scalar_mul(alpha, output)
    }
}

impl UnaryOperator for ScalarMul {
    fn forward(&self, input: &Tensor) -> Result<Tensor, crate::Error> {
        let input_t: &GenericTensor = &input.tensor().deref().borrow();
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

        {
            let inputs: &[&GenericTensor] = &[&inputs[0].gradient().deref().borrow()];
            let outputs: &[&GenericTensor] = &[&outputs[0].gradient().deref().borrow()];

            let input = inputs[0];
            let output_ = outputs[0];
            if output_.requires_grad() {
                output.push_instruction(gradient_instruction!(
                    OpCode::Add,
                    &[input, output_],
                    &[output_],
                ));
            }
        }

        Ok(output)
    }
}
