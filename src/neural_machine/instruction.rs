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
    #[cfg(debug_assertions)]
    file: String,
    #[cfg(debug_assertions)]
    line: u32,
    #[cfg(debug_assertions)]
    column: u32,
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! instruction {
    ( $opcode:expr, $inputs:expr, $outputs:expr, $category:expr, ) => {
        Instruction::new(
            $opcode,
            $inputs,
            $outputs,
            $category,
            file!(),
            line!(),
            column!(),
        )
    };
}
#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! instruction {
    ( $opcode:expr, $inputs:expr, $outputs:expr, $category:expr, ) => {
        Instruction::new($opcode, $inputs, $outputs, $category)
    };
}

impl Instruction {
    pub fn new(
        opcode: OpCode,
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
        category: Category,
        #[cfg(debug_assertions)] file: &str,
        #[cfg(debug_assertions)] line: u32,
        #[cfg(debug_assertions)] column: u32,
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

        Self {
            opcode,
            inputs: inputs.into(),
            outputs: outputs.into(),
            category,
            #[cfg(debug_assertions)]
            file: file.into(),
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column,
        }
    }

    #[cfg(debug_assertions)]
    pub fn file(&self) -> &String {
        &self.file
    }

    #[cfg(debug_assertions)]
    pub fn line(&self) -> u32 {
        self.line
    }

    #[cfg(debug_assertions)]
    pub fn column(&self) -> u32 {
        self.column
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
