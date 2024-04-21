use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{Accelerator, Error, Forward, OperatorTrait, Tape, Tensor};

pub struct Operator {
    accelerator: Rc<Accelerator>,
    tape: Rc<RefCell<Tape>>,
    variant: Rc<RefCell<Box<dyn OperatorTrait>>>,
}

impl Forward for Operator {
    fn forward(&mut self, input: &Rc<Tensor>) -> Result<Rc<Tensor>, Error> {
        let inputs = vec![input.clone()];
        self.forward_inputs(&inputs)
    }

    fn accelerator(&self) -> Rc<Accelerator> {
        self.accelerator.clone()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
    }
}

impl Operator {
    pub fn new(
        accelerator: Rc<Accelerator>,
        tape: Rc<RefCell<Tape>>,
        variant: Rc<RefCell<Box<dyn OperatorTrait>>>,
    ) -> Self {
        Self {
            accelerator,
            tape,
            variant,
        }
    }

    pub fn forward_inputs(&mut self, inputs: &Vec<Rc<Tensor>>) -> Result<Rc<Tensor>, Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        let output = variant.forward(self.accelerator.deref(), inputs)?;

        self.tape
            .deref()
            .borrow_mut()
            .push(self.variant.clone(), inputs.clone(), output.clone());
        Ok(output)
    }
}
