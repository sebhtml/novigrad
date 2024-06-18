use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

#[cfg(test)]
mod tests;

pub struct Transpose {}

impl Default for Transpose {
    fn default() -> Self {
        Self {}
    }
}

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
