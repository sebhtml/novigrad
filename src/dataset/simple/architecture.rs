use crate::{Device, Error, Forward, Operator, Operators, Tape, Tensor};
use std::{cell::RefCell, rc::Rc};

pub struct Architecture {
    embedding: Operator,
    linear_0: Operator,
    sigmoid_0: Operator,
    reshape: Operator,
    linear_1: Operator,
    sigmoid_1: Operator,
    linear_2: Operator,
    softmax: Operator,
}

impl Architecture {
    pub fn new(ops: &Operators) -> Self {
        Self {
            embedding: ops.embedding(16, 32),
            linear_0: ops.linear(16, 32, 6),
            sigmoid_0: ops.sigmoid(),
            reshape: ops.reshape(6, 16, 1, 6 * 16),
            linear_1: ops.linear(32, 6 * 16, 1),
            sigmoid_1: ops.sigmoid(),
            linear_2: ops.linear(16, 32, 1),
            softmax: ops.softmax(true),
        }
    }
}

impl Forward for Architecture {
    fn forward(&mut self, x: &Rc<Tensor>) -> Result<Rc<Tensor>, Error> {
        let x = self.embedding.forward(&x)?;
        let x = self.linear_0.forward(&x)?;
        let x = self.sigmoid_0.forward(&x)?;
        let x = self.reshape.forward(&x)?;
        let x = self.linear_1.forward(&x)?;
        let x = self.sigmoid_1.forward(&x)?;
        let x = self.linear_2.forward(&x)?;
        let x = self.softmax.forward(&x)?;
        Ok(x)
    }

    fn accelerator(&self) -> Rc<Device> {
        self.embedding.accelerator()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.embedding.tape()
    }
}
