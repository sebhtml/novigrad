use crate::{
    gradient_instruction, inference_instruction,
    tensor::{Error, Tensor},
    Device, DeviceInterface, OpCode, TensorWithGrad, UnaryOperator,
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

    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
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
        let input_t: &Tensor = &input.tensor().read().unwrap();
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
            &[&zero, &outputs[0].tensor().read().unwrap()],
            &[&outputs[0].tensor().read().unwrap()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].gradient().read().unwrap()],
            &[&outputs[0].gradient().read().unwrap()],
        ));
        let alpha = self.device.tensor(1, 1, vec![self.alpha])?;
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&alpha, &inputs[0].tensor().read().unwrap()],
            &[&outputs[0].tensor().read().unwrap()],
        ));
        let inputs = [&output];
        let outputs = [input];

        {
            let inputs: &[&Tensor] = &[&inputs[0].gradient().read().unwrap()];
            let outputs: &[&Tensor] = &[&outputs[0].gradient().read().unwrap()];

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
