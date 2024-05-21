use crate::{Error, GenericTensor};

pub struct Sub {}

impl Sub {
    pub fn execute(inputs: &[&GenericTensor], outputs: &[&GenericTensor]) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        GenericTensor::copy(input_0, output)?;
        GenericTensor::sub(input_1, output)
    }
}
