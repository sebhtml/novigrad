use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{Device, Error, Forward, LearningTensor, OperatorTrait, Tape};

pub struct Operator {
    device: Rc<Device>,
    tape: Rc<RefCell<Tape>>,
    variant: Rc<RefCell<Box<dyn OperatorTrait>>>,
}

impl Forward for Operator {
    fn forward(&self, inputs: &[LearningTensor]) -> Result<LearningTensor, Error> {
        let variant = &*self.variant.deref().borrow();
        let output = variant.forward(self.device.deref(), inputs)?;

        self.tape.deref().borrow_mut().push(
            self.variant.clone(),
            inputs.to_owned(),
            output.clone(),
        );

        Ok(output)
    }

    fn device(&self) -> Rc<Device> {
        self.device.clone()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
    }
}

impl Operator {
    pub fn new(
        device: Rc<Device>,
        tape: Rc<RefCell<Tape>>,
        variant: Rc<RefCell<Box<dyn OperatorTrait>>>,
    ) -> Self {
        Self {
            device,
            tape,
            variant,
        }
    }
}
