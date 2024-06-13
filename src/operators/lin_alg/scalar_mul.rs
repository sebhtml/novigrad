use crate::{
    gradient_instruction, inference_instruction, new_tensor, new_tensor_with_grad,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, OpCode, TensorWithGrad, UnaryOperator,
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
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let alpha = inputs[0];
        let input = inputs[1];
        let output = outputs[0];
        Tensor::copy(input, output)?;
        let device = input.device();
        device.scalar_mul(alpha, output)
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
        let alpha = new_tensor!(self.device, 1, 1, vec![self.alpha])?;
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&alpha, &inputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));
        let inputs = [&output];
        let outputs = [input];

        {
            let inputs: &[&Tensor] = &[&inputs[0].gradient()];
            let outputs: &[&Tensor] = &[&outputs[0].gradient()];

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
