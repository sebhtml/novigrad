use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{
    Accelerator, DeltaWorkingMemory, Error, Forward, Gradient, OperatorEnum, OperatorTrait, Tape,
    Tensor,
};

pub struct Operator {
    accelerator: Rc<Accelerator>,
    tape: Rc<RefCell<Tape>>,
    variant: Rc<RefCell<OperatorEnum>>,
}

impl Into<String> for Operator {
    fn into(self) -> String {
        let variant: &OperatorEnum = &self.variant.deref().borrow();
        variant.into()
    }
}

impl Forward for Operator {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, Error> {
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
        variant: Rc<RefCell<OperatorEnum>>,
    ) -> Self {
        Self {
            accelerator,
            tape,
            variant,
        }
    }

    pub fn forward_inputs(&mut self, inputs: &Vec<Tensor>) -> Result<Tensor, Error> {
        let mut output = Tensor::default();
        {
            let variant = &mut *self.variant.deref().borrow_mut();
            variant.forward(self.accelerator.deref(), inputs, &mut output)?;
        }

        self.tape
            .deref()
            .borrow_mut()
            .push(self.variant.clone(), inputs.clone(), output.clone());
        Ok(output)
    }

    pub fn compute_gradient(
        &mut self,
        inputs: &Vec<Tensor>,
        layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.compute_gradients(self.accelerator.deref(), inputs, layer_output_delta)
    }

    pub fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.commit_change(self.accelerator.deref(), learning_rate)
    }

    pub fn backward(
        &self,
        inputs: &Vec<Tensor>,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.backward(
            inputs,
            self.accelerator.deref(),
            layer_output_delta,
            previous_layer_output_delta,
        )
    }

    pub fn get_layer_output_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Tensor>,
        layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        layer_output_delta: &mut Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.get_layer_output_delta(
            self.accelerator.deref(),
            working_memory,
            inputs,
            layer_output,
            back_propagated_layer_output_delta,
            layer_output_delta,
        )
    }
}
