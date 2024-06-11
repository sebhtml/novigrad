use crate::devices::Device;
use crate::{
    emit_softmax_and_sigmoid_gradient_instructions, inference_instruction, tensor::Error,
    DeviceTrait, TensorWithGrad,
};
use crate::{new_tensor, new_tensor_with_grad};
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

    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _execution_unit: usize,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.sigmoid(input, output)
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
            OpCode::Sigmoid,
            &[&inputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));

        emit_softmax_and_sigmoid_gradient_instructions(&self.device, input, &output)?;

        Ok(output)
    }
}
