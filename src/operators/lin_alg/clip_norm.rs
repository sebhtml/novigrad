use crate::{Error, GenericTensor};

pub struct ClipNorm {}

impl ClipNorm {
    pub fn execute(
        clipped_norm: f32,
        inputs: &[&GenericTensor],
        outputs: &[&GenericTensor],
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            GenericTensor::copy(input, output)?;
        }
        output.clip(clipped_norm)?;
        Ok(())
    }
}
