use std::{cell::RefCell, rc::Rc};

use crate::{Device, Error, Forward, Operator, Operators, Tape, Tensor};

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
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let state_0 = self.embedding.forward(inputs)?;
        let state_1 = self.reshape.forward(&[state_0])?;
        let state_2 = self.linear.forward(&[state_1])?;
        let state_3 = self.softmax.forward(&[state_2])?;
        Ok(state_3)
    }

    fn device(&self) -> Rc<Device> {
        self.embedding.device()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.embedding.tape()
    }
}
