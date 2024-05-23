use crate::devices::Device;
use crate::{
    gradient_instruction, inference_instruction, CpuDevice, OpCode, Tensor, UnaryOperator,
};
use crate::{Error, TensorWithGrad};
use std::ops::Deref;

pub struct Softmax {
    device: Device,
    next_is_cross_entropy_loss: bool,
}

impl Softmax {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            next_is_cross_entropy_loss: false,
        }
    }

    pub fn new_with_next_is_cross_entropy_loss(device: &Device) -> Self {
        Self {
            device: device.clone(),
            next_is_cross_entropy_loss: true,
        }
    }

    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        /*
        debug_assert_eq!(false, input.is_nan()?,);
        let device = input.device();
        device.softmax(input, output)?;
        debug_assert_eq!(false, output.is_nan()?,);
        Ok(())
        */
        let input_values = input.get_values()?;
        let mut output_values = output.get_values()?;
        CpuDevice::_softmax(
            input.rows() as i32,
            input.cols() as i32,
            input_values.as_ptr(),
            output_values.as_mut_ptr(),
        )?;
        output.set_values(output_values)
    }
}

impl UnaryOperator for Softmax {
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
            emit_softmax_and_sigmoid_gradient_instructions(&self.device, input, &output)?;
        }

        Ok(output)
    }
}

pub fn emit_softmax_and_sigmoid_gradient_instructions(
    device: &Device,
    input: &TensorWithGrad,
    output: &TensorWithGrad,
) -> Result<(), Error> {
    let inputs = [&output];
    let outputs = [input];
    let inputs: &[&Tensor] = &[
        &inputs[0].tensor().deref().borrow(),
        &inputs[0].gradient().deref().borrow(),
        &outputs[0].tensor().deref().borrow(),
    ];
    let outputs: &[&Tensor] = &[&outputs[0].gradient().deref().borrow()];

    if outputs[0].requires_grad() {
        let output_gradient = outputs[0];
        let input_gradient = inputs[1];
        let output_ = inputs[2];
        let input = inputs[0];
        let rows = output_.rows();
        let cols = output_.cols();
        let len = rows * cols;
        let ones = device.tensor(rows, cols, vec![1.0; len])?;
        let one_minus_output = device.tensor(rows, cols, vec![0.0; len])?;

        output.push_instruction(gradient_instruction!(
            OpCode::Sub,
            &[&ones, input],
            &[&one_minus_output],
        ));
        let layer_f_derivative = device.tensor(rows, cols, vec![0.0; len])?;
        output.push_instruction(gradient_instruction!(
            OpCode::Mul,
            &[input, &one_minus_output],
            &[&layer_f_derivative],
        ));
        let tmp = device.tensor(rows, cols, vec![0.0; len])?;

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
    Ok(())
}
