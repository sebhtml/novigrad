use std::ops::Deref;

use crate::{
    devices::Device, gradient_instruction, loss_instruction, BinaryOperator, Error, ErrorEnum,
    GenericTensor, Instruction, OpCode, Tensor,
};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct ResidualSumOfSquares {
    device: Device,
}

impl ResidualSumOfSquares {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn execute(inputs: &[&GenericTensor], outputs: &[&GenericTensor]) -> Result<(), Error> {
        let expected = inputs[0];
        let actual = inputs[1];
        let loss = ResidualSumOfSquares::evaluate(expected, actual)?;
        outputs[0].set_values(vec![loss; 1]);
        Ok(())
    }

    /// RSS = Σ (y_i - f(x_i))^2
    fn evaluate(expected: &GenericTensor, actual: &GenericTensor) -> Result<f32, Error> {
        if expected.size() != actual.size() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }
        let expected_values = expected.get_values()?;
        let actual_values = actual.get_values()?;
        let mut loss = 0.0;
        for i in 0..expected_values.len() {
            let expected = expected_values[i];
            let actual = actual_values[i];
            let diff = expected - actual;
            loss += diff * diff;
        }
        Ok(loss)
        /*
        TODO use copy, sub, dot_product on GPU.
        TensorF32::copy(expected, diffs)?;
        TensorF32::sub(actual, diffs)?;
        TensorF32::dot_product(diffs, diffs)
         */
    }
}

impl BinaryOperator for ResidualSumOfSquares {
    fn forward(&self, input_1: &Tensor, input_2: &Tensor) -> Result<Tensor, Error> {
        let output = self
            .device
            .tensor(1, 1, vec![0.0], &[input_1, input_2], true, false);
        let inputs = [input_1, input_2];
        let outputs = [&output];
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul(0.0),
            &[&outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul(0.0),
            &[&outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::ResidualSumOfSquares,
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [input_1, input_2];
        let outputs = [input_2];
        let inputs: &[&GenericTensor] = &[
            &inputs[0].tensor().deref().borrow(),
            &inputs[1].tensor().deref().borrow(),
        ];
        let outputs: &[&GenericTensor] = &[&outputs[0].gradient().deref().borrow()];

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
            output.push_instruction(gradient_instruction!(
                OpCode::ScalarMul(-2.0),
                &[output_gradient],
                &[output_gradient],
            ));
        }

        Ok(output)
    }
}
