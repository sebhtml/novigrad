use std::{cell::RefCell, rc::Rc};

use crate::{Device, Error, Forward, Operator, Operators, Tape, Tensor};

pub struct Architecture {
    parameters: Tensor,
    embedding: Operator,
    matmul: Operator,
    reshape: Operator,
    linear: Operator,
    softmax: Operator,
}

impl Architecture {
    pub fn new(ops: &Operators, vocab_size: usize) -> Self {
        let device = ops.device();
        Self {
            parameters: device.tensor(384, 384, vec![0.0; 384 * 384], true),
            embedding: ops.embedding(vocab_size, 384),
            matmul: ops.matmul(),
            reshape: ops.reshape(32, 384, 1, 32 * 384),
            linear: ops.linear(vocab_size, 32 * 384, 1),
            softmax: ops.softmax(true),
        }
    }
}

impl Forward for Architecture {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        /*
        state_0 shape (32, 384)
        state_0b shape (32, 384)
        state_1 shape (1, 12288)
        state_2 shape (1, 34816)
        state_3 shape (1, 34816)
         */
        let state_0 = self.embedding.forward(inputs)?;
        let state_0b = self.matmul.forward(&[state_0, self.parameters.clone()])?;
        let state_1 = self.reshape.forward(&[state_0b])?;
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
