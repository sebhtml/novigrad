use crate::{Device, Error, OperatorTrait, TensorF32};
use core::fmt::Debug;
use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc};

#[derive(Clone, Debug)]
pub struct Tensor {
    device: Device,
    operator: Rc<dyn OperatorTrait>,
    inputs: Rc<Vec<Tensor>>,
    tensor: Rc<RefCell<TensorF32>>,
    gradient: Rc<RefCell<TensorF32>>,
    realized: Rc<RefCell<bool>>,
}

impl Tensor {
    pub fn new(
        device: &Device,
        operator: Rc<dyn OperatorTrait>,
        inputs: &[Tensor],
        tensor: Rc<RefCell<TensorF32>>,
        gradient: Rc<RefCell<TensorF32>>,
    ) -> Self {
        Self {
            device: device.clone(),
            operator,
            inputs: Rc::new(inputs.to_owned()),
            tensor,
            gradient,
            realized: Default::default(),
        }
    }

    pub fn operator(&self) -> &Rc<dyn OperatorTrait> {
        &self.operator
    }

    pub fn inputs(&self) -> &Vec<Tensor> {
        &self.inputs
    }

    pub fn realize(&self) -> Result<(), Error> {
        let realized = *self.realized.deref().borrow();
        debug_assert_eq!(realized, false);
        if realized {
            return Ok(());
        }
        let tape = self.get_tape();
        for output in tape.iter() {
            let op = output.operator();
            let inputs = output.inputs();
            op.forward_realize(inputs, output)?;
        }
        let realized: &mut bool = &mut self.realized.deref().borrow_mut();
        *realized = true;
        Ok(())
    }

    pub fn tensor(&self) -> &Rc<RefCell<TensorF32>> {
        &self.tensor
    }

    pub fn gradient(&self) -> &Rc<RefCell<TensorF32>> {
        &self.gradient
    }

    pub fn resize(&self, new_size: &[usize]) {
        self.tensor.deref().borrow_mut().reallocate(new_size);
        self.gradient.deref().borrow_mut().reallocate(new_size);
    }

    pub fn get_tape(&self) -> Vec<Tensor> {
        let mut tape = vec![];
        let mut stack = LinkedList::new();
        stack.push_back(self.clone());
        while let Some(element) = stack.pop_back() {
            if element.inputs.is_empty() {
                continue;
            }
            for input in element.inputs() {
                stack.push_back(input.clone());
            }
            tape.push(element);
        }
        tape.into_iter().rev().collect()
    }

    pub fn print_tape(tape: &[Tensor]) {
        println!("Tape");
        for (i, element) in tape.iter().enumerate() {
            println!(
                "index {}  operator {}  inputs {}",
                i,
                element.operator().name(),
                element.inputs().len()
            );
        }
    }

    /// Back-propagation
    pub fn backward(&self) -> Result<Rc<RefCell<Vec<Tensor>>>, Error> {
        let tape = self.get_tape();

        /*
            println!("----");
            Self::print_tape(&tape);
        */
        for output in tape.iter().rev() {
            let operator = output.operator().deref();
            let inputs = output.inputs();

            // Store enabled gradients to optimize them later.
            operator.backward(inputs, output)?;

            // Clip the backward gradients.
            for input in inputs {
                let backward_gradient: &mut TensorF32 = &mut input.gradient().deref().borrow_mut();
                let back_propagated_gradient = self.device.tensor_f32(
                    backward_gradient.rows(),
                    backward_gradient.cols(),
                    backward_gradient.get_values()?,
                );
                back_propagated_gradient.clip(-1.0, 1.0, backward_gradient)?;
            }
        }
        Ok(self.device.tensors_with_requires_grad().clone())
    }
}
