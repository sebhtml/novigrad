use crate::{tensor::Error, tensor::Tensor, DeviceTrait};

pub struct Clip {}

impl Clip {
    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let min = inputs[0];
        let max = inputs[1];
        let input = inputs[2];
        let output = outputs[0];
        let device = input.device();
        device.clip(min, max, input, output)?;
        Ok(())
    }
}
