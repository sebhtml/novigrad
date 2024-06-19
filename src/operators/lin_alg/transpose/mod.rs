use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

#[cfg(test)]
mod tests;

#[derive(Default)]
pub struct Transpose {}

impl ExecutableOperator for Transpose {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.transpose(input, output, device_stream)?;
        Ok(())
    }
}
