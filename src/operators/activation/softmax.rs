use crate::devices::Device;
use crate::{gradient_instruction, inference_instruction, GenericTensor, OpCode, UnaryOperator};
use crate::{Error, Tensor};
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

    pub fn execute(inputs: &[&GenericTensor], outputs: &[&GenericTensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        debug_assert_eq!(false, input.is_nan()?,);
        GenericTensor::softmax(input, output)?;
        debug_assert_eq!(false, output.is_nan()?,);
        Ok(())
    }
}

impl UnaryOperator for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
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
    let inputs: &[&GenericTensor] = &[
        &inputs[0].tensor().deref().borrow(),
        &inputs[0].gradient().deref().borrow(),
        &outputs[0].tensor().deref().borrow(),
    ];
    let outputs: &[&GenericTensor] = &[&outputs[0].gradient().deref().borrow()];

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
