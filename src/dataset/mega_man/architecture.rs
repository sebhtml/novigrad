use std::{cell::RefCell, rc::Rc};

use crate::{Accelerator, DifferentiableModule, Error, Forward, Operators, Tape, Tensor};

pub struct Architecture {
    embedding: DifferentiableModule,
    reshape: DifferentiableModule,
    linear: DifferentiableModule,
    softmax: DifferentiableModule,
}

impl Default for Architecture {
    fn default() -> Self {
        let ops = Operators::default();
        Self {
            embedding: ops.embedding(256, 384),
            reshape: ops.reshape(32, 384, 1, 32 * 384),
            linear: ops.linear(256, 32 * 384, 1),
            softmax: ops.softmax(true),
        }
    }
}

impl Forward for Architecture {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor, Error> {
        let x = self.embedding.forward(&x)?;
        let x = self.reshape.forward(&x)?;
        let x = self.linear.forward(&x)?;
        let x = self.softmax.forward(&x)?;
        Ok(x)
    }

    fn accelerator(&self) -> Rc<Accelerator> {
        self.embedding.accelerator()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.embedding.tape()
    }
}
