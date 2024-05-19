use std::rc::Rc;

use crate::{Error, OpCode, Operator, TensorF32};

#[derive(Clone, Debug, PartialEq)]
pub enum Category {
    Inference,
    Loss,
    Gradient,
    Optimization,
}

#[derive(Clone, Debug)]
pub struct Instruction {
    opcode: OpCode,
    inputs: Rc<Vec<TensorF32>>,
    outputs: Rc<Vec<TensorF32>>,
    category: Category,
}

impl Instruction {
    pub fn new(
        opcode: OpCode,
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
        category: Category,
    ) -> Self {
        let inputs: Vec<TensorF32> = inputs
            .to_owned()
            .into_iter()
            .map(|x| (*x).clone())
            .collect();
        let outputs: Vec<TensorF32> = outputs
            .to_owned()
            .into_iter()
            .map(|x| (*x).clone())
            .collect();
        println!("Instruction::new inputs {}", inputs.len());
        Self {
            opcode,
            inputs: inputs.into(),
            outputs: outputs.into(),
            category,
        }
    }

    pub fn category(&self) -> Category {
        self.category.clone()
    }

    pub fn opcode(&self) -> &OpCode {
        &self.opcode
    }
    pub fn inputs(&self) -> &Rc<Vec<TensorF32>> {
        &self.inputs
    }
    pub fn outputs(&self) -> &Rc<Vec<TensorF32>> {
        &self.outputs
    }
    pub fn forward(&self) -> Result<(), Error> {
        let inputs: Vec<&TensorF32> = self.inputs.iter().collect();
        let outputs_f32: Vec<&TensorF32> = self.outputs.iter().collect();
        self.opcode.forward(&inputs, &outputs_f32)
    }
}
