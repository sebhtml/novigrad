use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct ReduceSum {}

impl ExecutableOperator for ReduceSum {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        device.sum(input, output)
    }
}
