use crate::{tensor::Error, tensor::Tensor, Category, Instruction};
use core::fmt::Debug;
use std::fmt::Display;
use std::sync::{Arc, RwLock};
use std::{collections::LinkedList, ops::Deref};

#[derive(Clone, Debug)]
pub struct TensorWithGrad {
    inputs: Arc<Vec<TensorWithGrad>>,
    instructions: Arc<RwLock<Vec<Instruction>>>,
    tensor: Arc<RwLock<Tensor>>,
    gradient: Arc<RwLock<Tensor>>,
}

impl TensorWithGrad {
    pub fn new(tensor: Tensor, gradient: Tensor, inputs: &[&TensorWithGrad]) -> Self {
        let inputs: Vec<TensorWithGrad> = inputs.iter().map(|x| (*x).to_owned()).collect();
        Self {
            inputs: Arc::new(inputs),
            instructions: Default::default(),
            tensor: Arc::new(RwLock::new(tensor)),
            gradient: Arc::new(RwLock::new(gradient)),
        }
    }

    pub fn push_instruction(&self, instruction: Instruction) {
        self.instructions.write().unwrap().push(instruction)
    }

    pub fn forward_instructions(&self) -> Vec<Instruction> {
        self.instructions
            .read()
            .unwrap()
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Inference || i.category() == Category::Loss)
            .collect()
    }

    pub fn gradient_instructions(&self) -> Vec<Instruction> {
        self.instructions
            .read()
            .unwrap()
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Gradient)
            .collect()
    }

    pub fn tensor(&self) -> &Arc<RwLock<Tensor>> {
        &self.tensor
    }

    pub fn gradient(&self) -> &Arc<RwLock<Tensor>> {
        &self.gradient
    }

    pub fn get_tape(&self) -> Vec<TensorWithGrad> {
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

impl PartialEq for TensorWithGrad {
    fn eq(&self, other: &Self) -> bool {
        let t1: &Tensor = &self.tensor().read().unwrap();
        let t2: &Tensor = &other.tensor().read().unwrap();
        t1 == t2
    }
}

impl Display for TensorWithGrad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tensor: &Tensor = &self.tensor().read().unwrap();
        std::fmt::Display::fmt(&tensor, f)
    }
}
