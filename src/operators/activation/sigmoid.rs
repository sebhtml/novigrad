use crate::devices::Device;
use crate::opcode::OpCode;
use crate::stream::DeviceStream;
use crate::{
    emit_softmax_and_sigmoid_gradient_instructions, inference_instruction, tensor::Error,
    DeviceTrait, TensorWithGrad,
};
use crate::{new_tensor_with_grad, ExecutableOperator, OperatorAttributes};
use crate::{tensor::Tensor, UnaryOperator};

pub struct Sigmoid {
    device: Device,
}

impl Sigmoid {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl ExecutableOperator for Sigmoid {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.sigmoid(input, output, device_stream)
    }
}

impl UnaryOperator for Sigmoid {
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
        output.push_instruction(inference_instruction!(
            OpCode::Sigmoid,
            OperatorAttributes::None,
            &[&inputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));

        emit_softmax_and_sigmoid_gradient_instructions(&self.device, input, &output)?;

        Ok(output)
    }
}
