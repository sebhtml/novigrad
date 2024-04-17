use std::{cell::RefCell, rc::Rc};

use crate::{Accelerator, DifferentiableModule, Error, Forward, Session, Tape, Tensor};

pub struct Architecture {
    embedding: DifferentiableModule,
    reshape: DifferentiableModule,
    linear: DifferentiableModule,
    softmax: DifferentiableModule,
}

impl Default for Architecture {
    fn default() -> Self {
        let session = Session::default();
        let embedding = session.embedding(256, 384);
        let reshape = session.reshape(32, 384, 1, 32 * 384);
        let linear = session.linear(256, 32 * 384, 1);
        let softmax = session.softmax(true);
        Self {
            embedding,
            reshape,
            linear,
            softmax,
        }
    }
}

impl Forward for Architecture {
    fn forward(&mut self, layer_input: &Tensor) -> Result<Tensor, Error> {
        let embedding = self.embedding.forward(layer_input)?;
        let reshape = self.reshape.forward(&embedding)?;
        let linear = self.linear.forward(&reshape)?;
        let softmax = self.softmax.forward(&linear)?;
        Ok(softmax)
    }

    fn accelerator(&self) -> Rc<Accelerator> {
        self.embedding.accelerator()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.embedding.tape()
    }
}
