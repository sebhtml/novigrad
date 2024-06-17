use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct Copy {
}

impl ExecutableOperator for Copy {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.copy(
            input.len() as i32,
            input.as_mut_ptr(),
            1,
            output.as_mut_ptr(),
            1,
            device_stream,
        )
    }
}
