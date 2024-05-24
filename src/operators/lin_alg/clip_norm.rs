use crate::{Error, Tensor};

pub struct Normalize {}

impl Normalize {
    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            Tensor::copy(input, output)?;
        }
        output.normalize()?;
        Ok(())
    }
}
