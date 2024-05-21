use crate::{Category, Error, GenericTensor, Instruction};
use core::fmt::Debug;
use std::fmt::Display;
use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc};

#[derive(Clone, Debug)]
pub struct Tensor {
    inputs: Rc<Vec<Tensor>>,
    instructions: Rc<RefCell<Vec<Instruction>>>,
    tensor: Rc<RefCell<GenericTensor>>,
    gradient: Rc<RefCell<GenericTensor>>,
}

impl Tensor {
    pub fn new(tensor: GenericTensor, gradient: GenericTensor, inputs: &[&Tensor]) -> Self {
        let inputs: Vec<Tensor> = inputs.iter().map(|x| (*x).to_owned()).collect();
        Self {
            inputs: Rc::new(inputs),
            instructions: Default::default(),
            tensor: Rc::new(RefCell::new(tensor)),
            gradient: Rc::new(RefCell::new(gradient)),
        }
    }

    pub fn push_instruction(&self, instruction: Instruction) {
        self.instructions.deref().borrow_mut().push(instruction)
    }

    pub fn instructions(&self) -> Vec<Instruction> {
        self.instructions.deref().borrow().clone()
    }

    pub fn forward_instructions(&self) -> Vec<Instruction> {
        self.instructions()
            .into_iter()
            .filter(|i| i.category() == Category::Inference || i.category() == Category::Loss)
            .collect()
    }

    pub fn gradient_instructions(&self) -> Vec<Instruction> {
        self.instructions()
            .into_iter()
            .filter(|i| i.category() == Category::Gradient)
            .collect()
    }

    pub fn tensor(&self) -> &Rc<RefCell<GenericTensor>> {
        &self.tensor
    }

    pub fn gradient(&self) -> &Rc<RefCell<GenericTensor>> {
        &self.gradient
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
            inst.execute()?;
        }
        Ok(())
    }

    pub fn compute_gradient(&self) -> Result<(), Error> {
        for inst in self.gradient_instructions().iter() {
            inst.execute()?;
        }
        Ok(())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let t1: &GenericTensor = &self.tensor().deref().borrow();
        let t2: &GenericTensor = &other.tensor().deref().borrow();
        t1 == t2
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tensor: &GenericTensor = &self.tensor().deref().borrow();
        std::fmt::Display::fmt(&tensor, f)
    }
}
