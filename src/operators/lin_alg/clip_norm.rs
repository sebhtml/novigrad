use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    ExecutableOperator, OperatorAttributes,
};

pub struct ClipNorm {}

impl ExecutableOperator for ClipNorm {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            Tensor::copy(input, output, device_stream)?;
        }
        output.clip_norm(device_stream)?;
        Ok(())
    }
}
