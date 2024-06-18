use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct Bernoulli {}

impl ExecutableOperator for Bernoulli {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.bernoulli(input, output, device_stream)
    }
}
