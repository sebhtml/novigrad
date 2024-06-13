use crate::{
    devices::Device,
    gradient_instruction, loss_instruction, new_tensor, new_tensor_with_grad,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    BinaryOperator, DeviceTrait, OpCode, TensorWithGrad,
};

#[derive(Clone)]
pub struct SoftmaxCrossEntropyLoss {
    device: Device,
}

impl SoftmaxCrossEntropyLoss {
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
        let expected = inputs[0];
        let actual = inputs[1];
        let loss = outputs[0];
        let device = expected.device();
        device.cross_entropy_loss(expected, actual, loss)
    }
}

impl BinaryOperator for SoftmaxCrossEntropyLoss {
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
        let zero = new_tensor!(self.device, 1, 1, vec![0.0])?;
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].gradient()],
            &[&outputs[0].gradient()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::SoftmaxCrossEntropyLoss,
            &[&inputs[0].tensor(), &inputs[1].tensor(),],
            &[&outputs[0].tensor()],
        ));
        let inputs = [input_1, input_2];
        let outputs = [input_2];

        let inputs: &[&Tensor] = &[&inputs[0].tensor(), &inputs[1].tensor()];
        let outputs: &[&Tensor] = &[&outputs[0].gradient()];

        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);

        // When Cross-Entropy Loss is used with a Softmax activation function,
        // then we don't need to derive the softmax activations.
        // The derivative of the Loss in respect to logits (before activation) is
        // output of the softmax function - expected output (one-hot encoded)
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let expected = inputs[0];
            let actual = inputs[1];
            output.push_instruction(gradient_instruction!(
                OpCode::Sub,
                &[actual, expected],
                &[output_gradient],
            ));
        }

        Ok(output)
    }
}
