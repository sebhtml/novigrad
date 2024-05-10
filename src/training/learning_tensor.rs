use crate::{Device, Error, OperatorTrait, TensorF32};
use core::fmt::Debug;
use std::fmt::Display;
use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc};

#[derive(Clone, Debug)]
pub struct Tensor {
    device: Device,
    operator: Rc<dyn OperatorTrait>,
    inputs: Rc<Vec<Tensor>>,
    tensor: Rc<RefCell<TensorF32>>,
    gradient: Rc<RefCell<TensorF32>>,
    requires_grad: bool,
}

impl Tensor {
    pub fn new(
        device: &Device,
        operator: Rc<dyn OperatorTrait>,
        inputs: &[&Tensor],
        tensor: Rc<RefCell<TensorF32>>,
        gradient: Rc<RefCell<TensorF32>>,
        requires_grad: bool,
    ) -> Self {
        let inputs: Vec<Tensor> = inputs.into_iter().map(|x| (**x).clone()).collect();
        Self {
            device: device.clone(),
            operator,
            inputs: Rc::new(inputs.to_owned()),
            tensor,
            gradient,
            requires_grad,
        }
    }

    pub fn operator(&self) -> &Rc<dyn OperatorTrait> {
        &self.operator
    }

    pub fn inputs(&self) -> &Vec<Tensor> {
        &self.inputs
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn realize(&self) -> Result<(), Error> {
        let output = self;
        output.tensor().deref().borrow_mut().zero()?;
        output.gradient().deref().borrow_mut().zero()?;
        let op = output.operator();
        let inputs: Vec<_> = output.inputs().iter().collect();

        op.forward_realize(&inputs, output)?;
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
            let inputs: Vec<_> = output.inputs().iter().collect();

            operator.backward(&inputs, output)?;

            for input in inputs {
                if !input.requires_grad() {
                    continue;
                }
                let input_gradient: &mut TensorF32 = &mut input.gradient().deref().borrow_mut();
                let input_gradient_tmp = self.device.tensor_f32(
                    input_gradient.rows(),
                    input_gradient.cols(),
                    input_gradient.get_values()?,
                );
                // Clip the backward gradients.
                input_gradient_tmp.clip(-1.0, 1.0, input_gradient)?;
            }
        }
        Ok(self.device.tensors_with_requires_grad().clone())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let t1: &TensorF32 = &self.tensor().deref().borrow();
        let t2: &TensorF32 = &other.tensor().deref().borrow();
        t1 == t2
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tensor: &TensorF32 = &self.tensor().deref().borrow();
        std::fmt::Display::fmt(&tensor, f)
    }
}
