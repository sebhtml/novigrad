use crate::devices::Device;
use crate::{
    emit_softmax_and_sigmoid_gradient_instructions, inference_instruction, tensor::Error,
    DeviceInterface, TensorWithGrad,
};
use crate::{tensor::Tensor, OpCode, UnaryOperator};

pub struct Sigmoid {
    device: Device,
}

impl Sigmoid {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.sigmoid(input, output)
    }
}

impl UnaryOperator for Sigmoid {
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
        output.push_instruction(inference_instruction!(
            OpCode::Sigmoid,
            &[&inputs[0].tensor().read().unwrap()],
            &[&outputs[0].tensor().read().unwrap()],
        ));

        emit_softmax_and_sigmoid_gradient_instructions(&self.device, input, &output)?;

        Ok(output)
    }
}
