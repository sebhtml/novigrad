use std::ops::Deref;

use crate::{
    devices::Device, gradient_instruction, loss_instruction, BinaryOperator, DeviceInterface,
    Error, OpCode, Tensor, TensorWithGrad,
};

#[cfg(test)]
mod tests;

pub struct ReduceSumSquare {
    device: Device,
}

impl ReduceSumSquare {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let expected = inputs[0];
        let actual = inputs[1];
        let loss = outputs[0];
        let device = expected.device();
        device.reduce_square_sum(expected, actual, loss)
    }
}

impl BinaryOperator for ReduceSumSquare {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let output =
            self.device
                .tensor_with_grad(1, 1, vec![0.0], &[input_1, input_2], true, false)?;
        let inputs = [input_1, input_2];
        let outputs = [&output];
        let zero = self.device.tensor(1, 1, vec![0.0])?;
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::ReduceSumSquare,
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [input_1, input_2];
        let outputs = [input_2];
        let inputs: &[&Tensor] = &[
            &inputs[0].tensor().deref().borrow(),
            &inputs[1].tensor().deref().borrow(),
        ];
        let outputs: &[&Tensor] = &[&outputs[0].gradient().deref().borrow()];

        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let expected = inputs[0];
            let actual = inputs[1];
            output.push_instruction(gradient_instruction!(
                OpCode::Sub,
                &[expected, actual],
                &[output_gradient],
            ));
            let minus_two = self.device.tensor(1, 1, vec![-2.0])?;
            output.push_instruction(gradient_instruction!(
                OpCode::ScalarMul,
                &[&minus_two, output_gradient],
                &[output_gradient],
            ));
        }

        Ok(output)
    }
}
