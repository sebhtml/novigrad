use crate::devices::Device;
use crate::{
    gradient_instruction, inference_instruction, Instruction, OpCode, TensorF32, UnaryOperator,
};
use crate::{Error, Tensor};
use std::f32::consts::E;
use std::ops::Deref;

pub struct Softmax {
    device: Device,
    next_is_cross_entropy_loss: bool,
}

impl Softmax {
    pub fn new(device: &Device, next_is_cross_entropy_loss: bool) -> Self {
        Self {
            device: device.clone(),
            next_is_cross_entropy_loss,
        }
    }

    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        debug_assert_eq!(false, input.is_nan()?,);
        debug_assert_eq!(false, input.is_nan()?,);
        Self::activate(input, output)?;
        debug_assert_eq!(false, output.is_nan()?,);
        Ok(())
    }

    fn activate(input: &TensorF32, output: &TensorF32) -> Result<(), Error> {
        let rows = input.rows();
        let cols = input.cols();
        let values = input.get_values()?;
        let mut result_values = output.get_values()?;
        let mut row = 0;
        while row < rows {
            // Find max

            let mut max = values[input.index(row, 0)];
            let mut col = 0;
            while col < cols {
                let x = values[input.index(row, col)];
                max = max.max(x);
                col += 1;
            }

            // For each value:
            // 1. substract the max
            // 2. compute E^x
            // 3. add result to sum
            let mut sum = 0.0;
            let mut col = 0;
            while col < cols {
                let x = values[input.index(row, col)];
                debug_assert_eq!(false, x.is_nan());
                let y = E.powf(x - max);
                debug_assert_eq!(false, y.is_nan(), "x: {}, max: {}, y: {}", x, max, y,);
                result_values[output.index(row, col)] = y;
                sum += y;
                col += 1;
            }

            // Divide every value by sum.

            let mut col = 0;
            while col < cols {
                let x = result_values[output.index(row, col)];
                debug_assert_eq!(false, x.is_nan());
                debug_assert_ne!(0.0, sum);
                let y = x / sum;
                debug_assert_eq!(false, y.is_nan());
                result_values[output.index(row, col)] = y;
                col += 1;
            }
            row += 1;
        }
        output.set_values(result_values);
        Ok(())
    }
}

impl UnaryOperator for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
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
            OpCode::Softmax,
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));

        if self.next_is_cross_entropy_loss {
            output.push_instruction(gradient_instruction!(
                OpCode::Add,
                &[
                    &output.gradient().deref().borrow(),
                    &input.gradient().deref().borrow(),
                ],
                &[&input.gradient().deref().borrow()],
            ));
        } else {
            emit_softmax_and_sigmoid_gradient_instructions(&self.device, input, &output);
        }

        Ok(output)
    }
}

pub fn emit_softmax_and_sigmoid_gradient_instructions(
    device: &Device,
    input: &Tensor,
    output: &Tensor,
) {
    let inputs = [&output];
    let outputs = [input];
    let inputs: &[&TensorF32] = &[
        &inputs[0].tensor().deref().borrow(),
        &inputs[0].gradient().deref().borrow(),
        &outputs[0].tensor().deref().borrow(),
    ];
    let outputs: &[&TensorF32] = &[&outputs[0].gradient().deref().borrow()];

    if outputs[0].requires_grad() {
        let output_gradient = outputs[0];
        let input_gradient = inputs[1];
        let output_ = inputs[2];
        let input = inputs[0];
        let rows = output_.rows();
        let cols = output_.cols();
        let len = rows * cols;
        let ones = device.tensor_f32(rows, cols, vec![1.0; len]);
        let one_minus_output = device.tensor_f32(rows, cols, vec![0.0; len]);

        output.push_instruction(gradient_instruction!(
            OpCode::Sub,
            &[&ones, input],
            &[&one_minus_output],
        ));
        let layer_f_derivative = device.tensor_f32(rows, cols, vec![0.0; len]);
        output.push_instruction(gradient_instruction!(
            OpCode::Mul,
            &[input, &one_minus_output],
            &[&layer_f_derivative],
        ));
        let tmp = device.tensor_f32(rows, cols, vec![0.0; len]);

        output.push_instruction(gradient_instruction!(
            OpCode::Mul,
            &[&layer_f_derivative, input_gradient],
            &[&tmp],
        ));
        output.push_instruction(gradient_instruction!(
            OpCode::Add,
            &[&tmp, output_gradient],
            &[output_gradient],
        ));
    }
}
