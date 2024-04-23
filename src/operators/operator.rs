use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{Device, Error, Forward, OperatorTrait, Tape, Tensor};

pub struct Operator {
    device: Rc<Device>,
    tape: Rc<RefCell<Tape>>,
    variant: Rc<RefCell<Box<dyn OperatorTrait>>>,
}

impl Forward for Operator {
    fn forward(&mut self, input: &Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>, Error> {
        let inputs = vec![input.clone()];
        self.forward_inputs(&inputs)
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

    pub fn forward_inputs(
        &mut self,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
    ) -> Result<Rc<RefCell<Tensor>>, Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        let output = variant.forward(self.device.deref(), inputs)?;

        self.tape
            .deref()
            .borrow_mut()
            .push(self.variant.clone(), inputs.clone(), output.clone());
        Ok(output)
    }
}
