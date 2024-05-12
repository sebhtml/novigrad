use std::ops::Deref;

use crate::{Error, Operator, Tensor};

pub struct Clip {
    min: f32,
    max: f32,
}

impl Clip {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

impl Operator for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn forward(&self, _inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        for output in outputs {
            (*output)
                .gradient()
                .deref()
                .borrow_mut()
                .clip(self.min, self.max)?;
        }
        Ok(())
    }

    fn backward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        todo!()
    }
}
