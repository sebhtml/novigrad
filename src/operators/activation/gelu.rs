use crate::devices::Device;
use crate::opcode::OpCode;
use crate::stream::DeviceStream;
use crate::{
    gradient_instruction, new_tensor, new_tensor_with_grad, ExecutableOperator, OperatorAttributes,
};
use crate::{inference_instruction, tensor::Error, DeviceTrait, TensorWithGrad};
use crate::{tensor::Tensor, UnaryOperator};

/// https://onnx.ai/onnx/operators/onnx__Gelu.html
/// See
/// GAUSSIAN ERROR LINEAR UNITS (GELUS)
/// https://arxiv.org/pdf/1606.08415
pub struct Gelu {
    device: Device,
}

impl Gelu {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl ExecutableOperator for Gelu {
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

impl UnaryOperator for Gelu {
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
            OpCode::Gelu,
            OperatorAttributes::None,
            &[&inputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));

        output.push_instruction(gradient_instruction!(
            OpCode::Gelu,
            OperatorAttributes::None,
            &[&inputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));

        Ok(output)
    }
}
