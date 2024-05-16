use crate::{Error, Operator, TensorF32};

pub struct Clip {
    norm: f32,
}

impl Clip {
    pub fn new(norm: f32) -> Self {
        Self { norm }
    }
}

impl Operator for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn forward_f32(&self, _inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        for output in outputs {
            output.clip(self.norm)?;
        }
        Ok(())
    }
}
