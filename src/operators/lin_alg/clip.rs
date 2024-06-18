use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct Clip {}

impl ExecutableOperator for Clip {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let min = inputs[0];
        let max = inputs[1];
        let input = inputs[2];
        let output = outputs[0];
        device.clip(min, max, input, output, device_stream)
    }
}
