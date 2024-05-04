use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, Forward, OperatorTrait, Tensor};

pub struct Operator {
    device: Rc<Device>,
    variant: Rc<dyn OperatorTrait>,
}

impl Forward for Operator {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let variant = self.variant.deref();
        let output = variant.forward(self.device.deref(), inputs)?;

        Ok(output)
    }

    fn device(&self) -> Rc<Device> {
        self.device.clone()
    }
}

impl Operator {
    pub fn new(device: Rc<Device>, variant: Rc<dyn OperatorTrait>) -> Self {
        Self { device, variant }
    }
}
