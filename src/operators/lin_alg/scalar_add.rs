use crate::{tensor::Error, tensor::Tensor, DeviceInterface};

pub struct ScalarAdd {}

impl ScalarAdd {
    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let alpha = inputs[0];
        let input = inputs[1];
        let output = outputs[0];
        let device = input.device();
        Tensor::copy(input, output)?;
        device.scalar_add(alpha, output)
    }
}
