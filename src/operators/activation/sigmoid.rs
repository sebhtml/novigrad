use crate::devices::Device;
use crate::{
    emit_softmax_and_sigmoid_gradient_instructions, inference_instruction, Error, TensorWithGrad,
};
use crate::{OpCode, Tensor, UnaryOperator};
use std::f32::consts::E;
use std::ops::Deref;

#[derive(Clone)]
pub struct Sigmoid {
    device: Device,
}

impl Sigmoid {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];

        let rows = input.rows();
        let cols = input.cols();
        let values = input.get_values()?;
        let mut result_values = output.get_values()?;
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = values[input.index(row, col)];
                let y = 1.0 / (1.0 + E.powf(-x));
                result_values[output.index(row, col)] = y;
                col += 1;
            }
            row += 1;
        }
        output.set_values(result_values)
    }
}

impl UnaryOperator for Sigmoid {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let input_t: &Tensor = &input.tensor().deref().borrow();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output =
            self.device
                .tensor_with_grad(rows, cols, vec![0.0; len], &[input], true, false)?;

        let inputs = [input];
        let outputs = [&output];
        let zero = self.device.tensor(1, 1, vec![0.0])?;
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::Sigmoid,
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));

        emit_softmax_and_sigmoid_gradient_instructions(&self.device, input, &output)?;

        Ok(output)
    }
}
