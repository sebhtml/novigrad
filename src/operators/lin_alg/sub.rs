use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    ExecutableOperator, OperatorAttributes,
};

pub struct Sub {}

impl ExecutableOperator for Sub {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        Tensor::copy(input_0, output, device_stream)?;
        Tensor::sub(input_1, output, device_stream)
    }
}
