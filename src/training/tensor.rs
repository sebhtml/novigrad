use crate::{Error, Instruction, Operator, TensorF32};
use core::fmt::Debug;
use std::fmt::Display;
use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc};

#[derive(Clone, Debug)]
pub struct Tensor {
    inputs: Rc<Vec<Tensor>>,
    instructions: Rc<RefCell<Vec<Instruction>>>,
    tensor: Rc<RefCell<TensorF32>>,
    gradient: Rc<RefCell<TensorF32>>,
}

impl Tensor {
    pub fn new(tensor: TensorF32, gradient: TensorF32, inputs: &[&Tensor]) -> Self {
        let inputs: Vec<Tensor> = inputs.iter().map(|x| (*x).to_owned()).collect();
        Self {
            inputs: Rc::new(inputs),
            instructions: Default::default(),
            tensor: Rc::new(RefCell::new(tensor)),
            gradient: Rc::new(RefCell::new(gradient)),
        }
    }

    pub fn push_instruction(
        &self,
        operator: Rc<dyn Operator>,
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
        gradient_pathway: bool,
    ) {
        let instruction = Instruction::new(operator, inputs, outputs, gradient_pathway);
        self.instructions.deref().borrow_mut().push(instruction)
    }

    pub fn forward_instructions(&self) -> Vec<Instruction> {
        self.instructions
            .deref()
            .borrow()
            .clone()
            .into_iter()
            .filter(|i| !i.gradient_pathway())
            .collect()
    }

    pub fn backward_instructions(&self) -> Vec<Instruction> {
        self.instructions
            .deref()
            .borrow()
            .clone()
            .into_iter()
            .filter(|i| i.gradient_pathway())
            .collect()
    }

    pub fn requires_grad(&self) -> bool {
        self.gradient.deref().borrow().requires_grad()
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
            {
                let forward_instructions: Vec<Instruction> = element.forward_instructions();
                if forward_instructions.is_empty() {
                    continue;
                }
                let inputs = element.inputs.deref();
                for input in inputs.deref().iter() {
                    stack.push_back(input.clone());
                }
            }

            tape.push(element);
        }
        tape.into_iter().rev().collect()
    }

    pub fn forward(&self) -> Result<(), Error> {
        for inst in self.forward_instructions().iter() {
            inst.forward()?;
        }
        Ok(())
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
