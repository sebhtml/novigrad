use crate::devices::Device;
use crate::{
    gradient_instruction, inference_instruction, tensor::Tensor, DeviceTrait, OpCode, UnaryOperator,
};
use crate::{new_tensor, new_tensor_with_grad};
use crate::{tensor::Error, TensorWithGrad};

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

    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _execution_unit: usize,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.softmax(input, output)
    }
}

impl UnaryOperator for Softmax {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let input_t: &Tensor = &input.tensor();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output = new_tensor_with_grad!(
            self.device,
            rows,
            cols,
            vec![0.0; len],
            &[input],
            true,
            false
        )?;
        let inputs = [input];
        let outputs = [&output];
        let zero = new_tensor!(self.device, 1, 1, vec![0.0])?;
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].gradient()],
            &[&outputs[0].gradient()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::Softmax,
            &[&inputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));

        if self.next_is_cross_entropy_loss {
            output.push_instruction(gradient_instruction!(
                OpCode::Add,
                &[&output.gradient(), &input.gradient(),],
                &[&input.gradient()],
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
        &inputs[0].tensor(),
        &inputs[0].gradient(),
        &outputs[0].tensor(),
    ];
    let outputs: &[&Tensor] = &[&outputs[0].gradient()];

    if outputs[0].requires_grad() {
        let output_gradient = outputs[0];
        let input_gradient = inputs[1];
        let output_ = inputs[2];
        let input = inputs[0];
        let rows = output_.rows();
        let cols = output_.cols();
        let len = rows * cols;
        let ones = new_tensor!(device, rows, cols, vec![1.0; len])?;
        let one_minus_output = new_tensor!(device, rows, cols, vec![0.0; len])?;

        output.push_instruction(gradient_instruction!(
            OpCode::Sub,
            &[&ones, input],
            &[&one_minus_output],
        ));
        let layer_f_derivative = new_tensor!(device, rows, cols, vec![0.0; len])?;
        output.push_instruction(gradient_instruction!(
            OpCode::Mul,
            &[input, &one_minus_output],
            &[&layer_f_derivative],
        ));
        let tmp = new_tensor!(device, rows, cols, vec![0.0; len])?;

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
