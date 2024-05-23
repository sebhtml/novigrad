use crate::{Error, Tensor};

pub struct ClipNorm {}

impl ClipNorm {
    pub fn execute(
        clipped_norm: f32,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            Tensor::copy(input, output)?;
        }
        output.clip(clipped_norm)?;
        Ok(())
    }
}
