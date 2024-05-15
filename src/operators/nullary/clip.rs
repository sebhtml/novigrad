use std::ops::Deref;

use crate::{Error, Operator, Tensor, TensorF32};

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

    fn forward(&self, _inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let outputs: Vec<TensorF32> = outputs
            .iter()
            .map(|t| t.gradient().deref().borrow().clone())
            .collect();
        self.forward_f32(&[], &outputs.iter().collect::<Vec<_>>())
    }

    fn forward_f32(&self, _inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        for output in outputs {
            output.clip(self.norm)?;
        }
        Ok(())
    }
}
