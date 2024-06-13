use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    DeviceTrait,
};

pub struct ReduceSum {}

impl ReduceSum {
    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.sum(input, output)
    }
}
