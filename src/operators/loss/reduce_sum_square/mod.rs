use crate::{
    devices::Device,
    gradient_instruction, loss_instruction, new_tensor, new_tensor_with_grad,
    opcode::OpCode,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    BinaryOperator, DeviceTrait, ExecutableOperator, OperatorAttributes, TensorWithGrad,
};

#[cfg(test)]
mod tests;

pub struct ReduceSumSquare {
    device: Device,
}

impl ReduceSumSquare {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl ExecutableOperator for ReduceSumSquare {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let expected = inputs[0];
        let actual = inputs[1];
        let loss = outputs[0];
        device.reduce_sum_square(expected, actual, loss, device_stream)
    }
}

impl BinaryOperator for ReduceSumSquare {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let output = new_tensor_with_grad!(
            self.device,
            1,
            1,
            vec![0.0],
            &[input_1, input_2],
            true,
            false
        )?;
        let inputs = [input_1, input_2];
        let outputs = [&output];

        output.push_instruction(loss_instruction!(
            OpCode::ReduceSumSquare,
            OperatorAttributes::None,
            &[&inputs[0].tensor(), &inputs[1].tensor(),],
            &[&outputs[0].tensor()],
        ));
        let inputs = [input_1, input_2];
        let outputs = [input_2];
        let inputs: &[&Tensor] = &[&inputs[0].tensor(), &inputs[1].tensor()];
        let outputs: &[&Tensor] = &[&outputs[0].gradient()];

        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let expected = inputs[0];
            let actual = inputs[1];
            output.push_instruction(gradient_instruction!(
                OpCode::Sub,
                OperatorAttributes::None,
                &[expected, actual],
                &[output_gradient],
            ));
            let minus_two = new_tensor!(self.device, 1, 1, vec![-2.0])?;
            output.push_instruction(gradient_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&minus_two, output_gradient],
                &[output_gradient],
            ));
        }

        Ok(output)
    }
}
