use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc};

use crate::{Device, Error, OperatorTrait, TensorF32};

#[derive(Clone)]
pub struct Tensor {
    operator: Rc<dyn OperatorTrait>,
    inputs: Rc<RefCell<Vec<Tensor>>>,
    tensor: Rc<RefCell<TensorF32>>,
    gradient: Rc<RefCell<TensorF32>>,
}

impl Tensor {
    pub fn new(
        operator: Rc<dyn OperatorTrait>,
        inputs: &[Tensor],
        tensor: Rc<RefCell<TensorF32>>,
        gradient: Rc<RefCell<TensorF32>>,
    ) -> Self {
        Self {
            operator,
            inputs: Rc::new(RefCell::new(inputs.to_owned())),
            tensor,
            gradient,
        }
    }

    pub fn set_operator(&mut self, operator: Rc<dyn OperatorTrait>) {
        self.operator = operator;
    }

    pub fn operator(&self) -> &Rc<dyn OperatorTrait> {
        &self.operator
    }

    pub fn set_inputs(&self, inputs: &[Tensor]) {
        let self_input: &mut Vec<Tensor> = &mut self.inputs.deref().borrow_mut();
        *self_input = inputs.to_owned();
    }

    pub fn inputs(&self) -> &Rc<RefCell<Vec<Tensor>>> {
        &self.inputs
    }

    pub fn tensor(&self) -> &Rc<RefCell<TensorF32>> {
        &self.tensor
    }

    pub fn gradient(&self) -> &Rc<RefCell<TensorF32>> {
        &self.gradient
    }

    pub fn shape(&self) -> (usize, usize) {
        self.tensor.deref().borrow().shape()
    }

    pub fn resize(&self, rows: usize, cols: usize) {
        self.tensor.deref().borrow_mut().resize(rows, cols);
        self.gradient.deref().borrow_mut().resize(rows, cols);
    }

    pub fn get_tape(&self) -> Vec<Tensor> {
        let mut tape = vec![];
        let mut stack = LinkedList::new();
        stack.push_back(self.clone());
        while let Some(element) = stack.pop_back() {
            {
                let inputs: &[Tensor] = &element.inputs().deref().borrow();
                for input in inputs {
                    stack.push_back(input.clone());
                }
            }
            tape.push(element);
        }
        tape.into_iter().rev().collect()
    }

    pub fn print_tape(tape: &[Tensor]) {
        println!("Tape");
        for (i, element) in tape.iter().enumerate() {
            let inputs: &[Tensor] = &element.inputs().deref().borrow();
            println!(
                "index {}  operator {}  inputs {}",
                i,
                element.operator().name(),
                inputs.len(),
            );
        }
    }

    /// Back-propagation
    pub fn backward(&self, device: &Device) -> Result<Vec<Tensor>, Error> {
        let tape = self.get_tape();

        /*
            println!("----");
            Self::print_tape(&tape);
        */
        for output in tape.iter().rev() {
            let operator = output.operator().deref();
            let inputs: &[Tensor] = &output.inputs().deref().borrow();
            if inputs.is_empty() {
                continue;
            }

            // Store enabled gradients to optimize them later.
            operator.backward(device, inputs, output)?;

            // Clip the backward gradients.
            for input in inputs {
                let backward_gradient: &mut TensorF32 = &mut input.gradient().deref().borrow_mut();
                let back_propagated_gradient = device.tensor_f32(
                    backward_gradient.rows(),
                    backward_gradient.cols(),
                    backward_gradient.get_values()?,
                );
                back_propagated_gradient.clip(-1.0, 1.0, backward_gradient)?;
            }
        }
        Ok(device.tensors_with_requires_grad())
    }
}
