use crate::{
    gradient_instruction, inference_instruction, new_tensor_with_grad,
    opcode::OpCode,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes, TensorWithGrad, UnaryOperator,
};

pub struct Standardization {
    device: Device,
}

impl Standardization {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl UnaryOperator for Standardization {
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
        output.push_instruction(inference_instruction!(
            OpCode::Standardization,
            OperatorAttributes::None,
            &[&input.tensor()],
            &[&output.tensor()],
        ));
        output.push_instruction(gradient_instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&output.gradient(), &input.gradient(),],
            &[&input.gradient()],
        ));
        Ok(output)
    }
}

impl ExecutableOperator for Standardization {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.standardization(input, output, device_stream)
    }
}
