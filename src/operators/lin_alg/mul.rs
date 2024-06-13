use crate::{
    gradient_instruction, inference_instruction, new_tensor, new_tensor_with_grad,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    BinaryOperator, Device, OpCode, TensorWithGrad,
};

pub struct Mul {
    device: Device,
}

impl Mul {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        Tensor::mul(input_0, input_1, output)
    }
}

impl BinaryOperator for Mul {
    fn forward(
        &self,
        input_0: &TensorWithGrad,
        input_1: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let input_0_t: &Tensor = &input_0.tensor();
        let input_1_t: &Tensor = &input_1.tensor();
        debug_assert_eq!(*input_0_t.size(), *input_1_t.size());
        let rows = input_0_t.rows();
        let cols = input_0_t.cols();
        let len = rows * cols;
        let output = new_tensor_with_grad!(
            self.device,
            rows,
            cols,
            vec![0.0; len],
            &[input_0, input_1],
            true,
            false,
        )?;
        let inputs = [input_0, input_1];
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
            OpCode::Mul,
            &[&inputs[0].tensor(), &inputs[1].tensor(),],
            &[&outputs[0].tensor()],
        ));

        {
            let inputs = [input_0, input_1, &output];
            let outputs = [input_0, input_1];

            let inputs: &[&Tensor] = &[
                &inputs[0].tensor(),
                &inputs[1].tensor(),
                &inputs[2].gradient(),
            ];

            let outputs = &[&outputs[0].gradient(), &outputs[1].gradient()];

            debug_assert_eq!(outputs.len(), 2);
            let input_gradient = inputs[2];
            let rows = input_gradient.rows();
            let cols = input_gradient.cols();
            let len = rows * cols;

            if outputs[1].requires_grad() {
                let output_1_gradient = outputs[1];
                let output_0 = inputs[0];
                let tmp = new_tensor!(self.device, rows, cols, vec![0.0; len])?;

                output.push_instruction(gradient_instruction!(
                    OpCode::Mul,
                    &[output_0, input_gradient],
                    &[&tmp],
                ));

                output.push_instruction(gradient_instruction!(
                    OpCode::Add,
                    &[&tmp, output_1_gradient],
                    &[output_1_gradient],
                ));
            }

            if outputs[0].requires_grad() {
                let output_0_gradient = outputs[0];
                let output_ = inputs[1];
                let tmp = new_tensor!(self.device, rows, cols, vec![0.0; len])?;

                output.push_instruction(gradient_instruction!(
                    OpCode::Mul,
                    &[output_, input_gradient],
                    &[&tmp],
                ));

                output.push_instruction(gradient_instruction!(
                    OpCode::Add,
                    &[&tmp, output_0_gradient],
                    &[output_0_gradient],
                ));
            }
        }

        Ok(output)
    }
}
