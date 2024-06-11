use crate::{tensor::Error, tensor::Tensor};

pub struct ClipNorm {}

impl ClipNorm {
    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _execution_unit: usize,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            Tensor::copy(input, output)?;
        }
        output.clip_norm()?;
        Ok(())
    }
}
