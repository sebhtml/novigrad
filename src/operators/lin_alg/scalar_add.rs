use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct ScalarAdd {}

impl ExecutableOperator for ScalarAdd {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let alpha = inputs[0];
        let input = inputs[1];
        let output = outputs[0];
        let device = input.device();
        Tensor::copy(input, output, device_stream)?;
        device.scalar_add(alpha, output)
    }
}
