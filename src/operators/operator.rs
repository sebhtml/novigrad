use std::{ops::Deref, rc::Rc};

use crate::{Error, Forward, OperatorTrait, Tensor};

pub struct Operator {
    variant: Rc<dyn OperatorTrait>,
}

impl Forward for Operator {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let variant = self.variant.deref();
        let output = variant.forward(inputs)?;

        Ok(output)
    }
}

impl Operator {
    pub fn new(variant: Rc<dyn OperatorTrait>) -> Self {
        Self { variant }
    }
}
