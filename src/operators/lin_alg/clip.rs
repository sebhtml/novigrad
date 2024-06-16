use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    DeviceTrait,
};

pub struct Clip {}

impl Clip {
    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let min = inputs[0];
        let max = inputs[1];
        let input = inputs[2];
        let output = outputs[0];
        let device = input.device();
        device.clip(min, max, input, output, device_stream)?;
        Ok(())
    }
}
