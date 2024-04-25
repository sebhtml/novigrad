use std::{cell::RefCell, rc::Rc};

use crate::{Device, Error, Forward, LearningTensor, Operator, Operators, Tape};

pub struct Architecture {
    embedding: Operator,
    reshape: Operator,
    linear: Operator,
    softmax: Operator,
}

impl Architecture {
    pub fn new(ops: &Operators) -> Self {
        Self {
            embedding: ops.embedding(256, 384),
            reshape: ops.reshape(32, 384, 1, 32 * 384),
            linear: ops.linear(256, 32 * 384, 1),
            softmax: ops.softmax(true),
        }
    }
}

impl Forward for Architecture {
    fn forward(&mut self, x: &LearningTensor) -> Result<LearningTensor, Error> {
        let x = self.embedding.forward(&x)?;
        let x = self.reshape.forward(&x)?;
        let x = self.linear.forward(&x)?;
        let x = self.softmax.forward(&x)?;
        Ok(x)
    }

    fn device(&self) -> Rc<Device> {
        self.embedding.device()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.embedding.tape()
    }
}
