use std::{ops::Deref, rc::Rc};

use crate::{Error, OperatorTrait, Tensor};

pub struct Operator {
    variant: Rc<dyn OperatorTrait>,
}

impl OperatorTrait for Operator {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let variant = self.variant.deref();
        let output = variant.forward(inputs)?;
        Ok(output)
    }

    fn name(&self) -> &str {
        "OperatorTrait"
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::UnsupportedOperation)
    }
}

impl Operator {
    pub fn new(variant: Rc<dyn OperatorTrait>) -> Self {
        Self { variant }
    }
}
