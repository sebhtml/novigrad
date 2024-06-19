use crate::{
    devices::Device,
    error, gradient_instruction, inference_instruction, new_tensor, new_tensor_with_grad,
    opcode::OpCode,
    stream::DeviceStream,
    tensor::{Error, ErrorEnum, Tensor},
    ExecutableOperator, OperatorAttributes, TensorWithGrad, UnaryOperator,
};

pub struct Reshape {
    device: Device,
    input_size: Vec<usize>,
    output_size: Vec<usize>,
}

impl Reshape {
    pub fn new(device: &Device, input_size: Vec<usize>, output_size: Vec<usize>) -> Self {
        Self {
            device: device.clone(),
            input_size,
            output_size,
        }
    }
}

impl ExecutableOperator for Reshape {
    fn execute(
        attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let output_size = match attributes {
            OperatorAttributes::Vec(output_size) => output_size,
            _ => {
                return Err(error!(ErrorEnum::UnsupportedOperation));
            }
        };
        let input = inputs[0];
        let output = outputs[0];
        device.copy_to(input, output, device_stream)?;
        output.resize(output_size)
    }
}

impl UnaryOperator for Reshape {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let input_tensor: &Tensor = &input.tensor();
        debug_assert_eq!(*input_tensor.size(), self.input_size);
        let rows = self.output_size[0];
        let cols = self.output_size[1];
        let len = rows * cols;
        let output = new_tensor_with_grad!(
            self.device,
            rows,
            cols,
            vec![0.0; len],
            &[input],
            true,
            false
        )
        .unwrap();
        let inputs = [input];
        let outputs = [&output];
        let zero = new_tensor!(self.device, 1, 1, vec![0.0]).unwrap();
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&zero, &outputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&zero, &outputs[0].gradient()],
            &[&outputs[0].gradient()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::Reshape,
            OperatorAttributes::Vec(self.output_size.clone()),
            &[&inputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));
        let inputs = [&output];
        let outputs = [input];

        if outputs[0].gradient().requires_grad() {
            output.push_instruction(gradient_instruction!(
                OpCode::Reshape,
                OperatorAttributes::Vec(self.input_size.clone()),
                &[&inputs[0].gradient()],
                &[&outputs[0].gradient()],
            ));
        }

        Ok(output)
    }
}
