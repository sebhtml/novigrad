use crate::{Error, Tensor};

pub struct ScalarAdd {}

impl ScalarAdd {
    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let alpha = inputs[0];
        let input = inputs[1];
        let output = outputs[0];
        Tensor::copy(input, output)?;
        Tensor::scalar_add(alpha, output)
    }
}
