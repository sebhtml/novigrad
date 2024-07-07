use crate::{
    instruction, new_tensor, new_tensor_with_grad,
    opcode::OpCode,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Category, Device, DeviceTrait, ExecutableOperator, OperatorAttributes, TensorWithGrad,
    UnaryOperator,
};

#[cfg(test)]
mod tests;

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
}

impl ExecutableOperator for ScalarMul {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let alpha = inputs[0];
        let input = inputs[1];
        let output = outputs[0];
        device.copy_to(input, output, device_stream)?;
        device.scalar_mul(alpha, output, device_stream)
    }
}

impl UnaryOperator for ScalarMul {
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

        let alpha = new_tensor!(self.device, 1, 1, vec![self.alpha])?;
        output.push_instruction(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&alpha, &inputs[0].tensor()],
            &[&outputs[0].tensor()],
            Category::Inference,
        ));
        let inputs = [&output];
        let outputs = [input];

        {
            let inputs: &[&Tensor] = &[&inputs[0].gradient()];
            let outputs: &[&Tensor] = &[&outputs[0].gradient()];

            let input = inputs[0];
            let output_ = outputs[0];
            if output_.requires_grad() {
                output.push_instruction(instruction!(
                    OpCode::Add,
                    OperatorAttributes::None,
                    &[input, output_],
                    &[output_],
                    Category::Gradient,
                ));
            }
        }

        Ok(output)
    }
}
