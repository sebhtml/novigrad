use crate::{
    devices::Device,
    instruction, new_tensor_with_grad,
    opcode::OpCode,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    BinaryOperator, Category, DeviceTrait, ExecutableOperator, OperatorAttributes, TensorWithGrad,
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
}

impl ExecutableOperator for SoftmaxCrossEntropyLoss {
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
        device.cross_entropy_loss(expected, actual, loss, device_stream)
    }
}

impl BinaryOperator for SoftmaxCrossEntropyLoss {
    fn forward(
        &self,
        expected: &TensorWithGrad,
        actual: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let output = new_tensor_with_grad!(
            self.device,
            1,
            1,
            vec![0.0],
            &[expected, actual],
            true,
            false
        )?;

        output.push_instruction(instruction!(
            OpCode::SoftmaxCrossEntropyLoss,
            OperatorAttributes::None,
            &[&expected.tensor(), &actual.tensor(),],
            &[&output.tensor()],
            Category::Loss,
        ));

        // When Cross-Entropy Loss is used with a Softmax activation function,
        // then we don't need to derive the softmax activations.
        // The derivative of the Loss in respect to logits (before activation) is
        // output of the softmax function - expected output (one-hot encoded)
        if actual.gradient().requires_grad() {
            output.push_instruction(instruction!(
                OpCode::Sub,
                OperatorAttributes::None,
                &[&actual.tensor(), &expected.tensor()],
                &[&actual.gradient()],
                Category::Gradient,
            ));
        }

        Ok(output)
    }
}
