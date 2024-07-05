use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct Dot {}

impl Default for Dot {
    fn default() -> Self {
        Self {}
    }
}

impl ExecutableOperator for Dot {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let lhs = inputs[0];
        let rhs = inputs[1];
        let dot = outputs[0];
        device.dot(lhs, rhs, dot, device_stream)
    }
}
