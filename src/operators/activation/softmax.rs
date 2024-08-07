use crate::devices::Device;
use crate::opcode::OpCode;
use crate::stream::DeviceStream;
use crate::{
    instruction, new_tensor, new_tensor_with_grad, Category, ExecutableOperator, OperatorAttributes,
};
use crate::{tensor::Error, TensorWithGrad};
use crate::{tensor::Tensor, DeviceTrait, UnaryOperator};

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
}

impl ExecutableOperator for Softmax {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.softmax(input, output, device_stream)
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

        output.push_instruction(instruction!(
            OpCode::Softmax,
            OperatorAttributes::None,
            &[&inputs[0].tensor()],
            &[&outputs[0].tensor()],
            Category::Inference,
        ));

        if input.gradient().requires_grad() {
            if self.next_is_cross_entropy_loss {
                output.push_instruction(instruction!(
                    OpCode::Add,
                    OperatorAttributes::None,
                    &[&output.gradient(), &input.gradient(),],
                    &[&input.gradient()],
                    Category::Gradient,
                ));
            } else {
                emit_softmax_and_sigmoid_gradient_instructions(&self.device, input, &output)?;
            }
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

        output.push_instruction(instruction!(
            OpCode::Sub,
            OperatorAttributes::None,
            &[&ones, input],
            &[&one_minus_output],
            Category::Gradient,
        ));
        let layer_f_derivative = new_tensor!(device, rows, cols, vec![0.0; len])?;
        output.push_instruction(instruction!(
            OpCode::Mul,
            OperatorAttributes::None,
            &[input, &one_minus_output],
            &[&layer_f_derivative],
            Category::Gradient,
        ));
        let tmp = new_tensor!(device, rows, cols, vec![0.0; len])?;

        output.push_instruction(instruction!(
            OpCode::Mul,
            OperatorAttributes::None,
            &[&layer_f_derivative, input_gradient],
            &[&tmp],
            Category::Gradient,
        ));
        output.push_instruction(instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&tmp, output_gradient],
            &[output_gradient],
            Category::Gradient,
        ));
    }
    Ok(())
}
