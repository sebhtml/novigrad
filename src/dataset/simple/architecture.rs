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
            embedding: ops.embedding(256, 32),
            linear_0: ops.linear(256, 32, 6),
            sigmoid_0: ops.sigmoid(),
            reshape: ops.reshape(6, 256, 1, 6 * 256),
            linear_1: ops.linear(256, 6 * 256, 1),
            sigmoid_1: ops.sigmoid(),
            linear_2: ops.linear(256, 256, 1),
            softmax: ops.softmax(true),
        }
    }
}

impl Forward for Architecture {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        /*
        state_0 shape (6, 32)
        state_1 shape (6, 256)
        state_2 shape (6, 256)
        state_3 shape (1, 1536)
        state_4 shape (1, 256)
        state_5 shape (1, 256)
        state_6 shape (1, 256)
        state_7 shape (1, 256)
         */
        let state_0: Tensor = self.embedding.forward(inputs)?;
        let state_1 = self.linear_0.forward(&[state_0])?;
        let state_2 = self.sigmoid_0.forward(&[state_1])?;
        let state_3 = self.reshape.forward(&[state_2])?;
        let state_4 = self.linear_1.forward(&[state_3])?;
        let state_5 = self.sigmoid_1.forward(&[state_4])?;
        let state_6 = self.linear_2.forward(&[state_5])?;
        let state_7 = self.softmax.forward(&[state_6])?;
        Ok(state_7)
    }

    fn device(&self) -> Rc<Device> {
        self.embedding.device()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.embedding.tape()
    }
}
