use crate::{
    error,
    stream::DeviceStream,
    tensor::{Error, ErrorEnum, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct Pow {}

impl Pow {
    pub fn new() -> Self {
        Self {}
    }
}

impl ExecutableOperator for Pow {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        if *input_0.size() != *input_1.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        if *input_0.size() != *output.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        device.pow(input_0, input_1, output, device_stream)?;
        Ok(())
    }
}
