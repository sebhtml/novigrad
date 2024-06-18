use crate::{
    gradient_instruction, inference_instruction, new_tensor_with_grad,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, ExecutableOperator, OpCode, OperatorAttributes, TensorWithGrad, UnaryOperator,
};

pub struct Identity {
    label: String,
}

impl Identity {
    pub fn new(label: String) -> Self {
        Self { label }
    }
}

impl ExecutableOperator for Identity {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.copy_to(input, output, device_stream)
    }
}

impl UnaryOperator for Identity {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let output = new_tensor_with_grad!(
            input.device(),
            input.tensor().rows(),
            input.tensor().cols(),
            vec![0.0; input.tensor().len()],
            &[input],
            true,
            false,
        )?;
        output.push_instruction(inference_instruction!(
            OpCode::Identity,
            OperatorAttributes::String(self.label.clone()),
            &[&input.tensor()],
            &[&output.tensor()],
        ));
        output.push_instruction(gradient_instruction!(
            OpCode::Identity,
            OperatorAttributes::String("gradient".into()),
            &[&output.gradient()],
            &[&input.gradient()],
        ));
        Ok(output)
    }
}
