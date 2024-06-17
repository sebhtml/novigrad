use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct ReduceL2 {}

impl ExecutableOperator for ReduceL2 {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.dot(input, input, output, device_stream)?;
        device.sqrt(output, output, device_stream)?;
        Ok(())
    }
}
