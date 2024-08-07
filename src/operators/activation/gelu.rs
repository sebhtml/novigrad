use crate::devices::Device;
use crate::opcode::OpCode;
use crate::stream::DeviceStream;
use crate::{
    instruction, new_tensor, new_tensor_with_grad, Category, ExecutableOperator, OperatorAttributes,
};
use crate::{tensor::Error, DeviceTrait, TensorWithGrad};
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
        device.gelu(input, output, device_stream)
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

        output.push_instruction(instruction!(
            OpCode::Gelu,
            OperatorAttributes::None,
            &[&input.tensor()],
            &[&output.tensor()],
            Category::Inference,
        ));

        if input.gradient().requires_grad() {
            let device = &self.device;
            let layer_f_derivative = new_tensor!(device, rows, cols, vec![0.0; len])?;
            output.push_instruction(instruction!(
                OpCode::GeluDerivative,
                OperatorAttributes::None,
                &[&input.tensor()],
                &[&layer_f_derivative],
                Category::Gradient,
            ));
            let tmp = new_tensor!(device, rows, cols, vec![0.0; len])?;
            output.push_instruction(instruction!(
                OpCode::Mul,
                OperatorAttributes::None,
                &[&output.gradient(), &layer_f_derivative],
                &[&tmp],
                Category::Gradient,
            ));
            output.push_instruction(instruction!(
                OpCode::Add,
                OperatorAttributes::None,
                &[&tmp, &input.gradient()],
                &[&input.gradient()],
                Category::Gradient,
            ));
        }

        Ok(output)
    }
}

pub struct GeluDerivative {}

impl ExecutableOperator for GeluDerivative {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.gelu_derivative(input, output, device_stream)
    }
}
