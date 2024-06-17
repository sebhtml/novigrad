use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    DeviceTrait,
};

pub struct ReduceL2 {}

impl ReduceL2 {
    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.dot(input, input, output, device_stream)?;
        device.sqrt(output, output)?;
        Ok(())
    }
}
