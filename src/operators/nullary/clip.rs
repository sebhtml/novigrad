use crate::{Error, TensorF32};

pub struct Clip {}

impl Clip {
    pub fn execute(
        clipped_norm: f32,
        _inputs: &[&TensorF32],
        outputs: &[&TensorF32],
    ) -> Result<(), Error> {
        for output in outputs {
            output.clip(clipped_norm)?;
        }
        Ok(())
    }
}
