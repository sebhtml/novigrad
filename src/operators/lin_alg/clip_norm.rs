use crate::{Error, TensorF32};

pub struct ClipNorm {}

impl ClipNorm {
    pub fn execute(
        clipped_norm: f32,
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            TensorF32::copy(input, output)?;
        }
        output.clip(clipped_norm)?;
        Ok(())
    }
}
