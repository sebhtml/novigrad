use crate::{Error, GenericTensor, OpCode};
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
pub enum Category {
    Inference,
    Loss,
    Gradient,
    Optimization,
}

impl Into<String> for Category {
    fn into(self) -> String {
        match self {
            Category::Inference => "Inference",
            Category::Loss => "Loss",
            Category::Gradient => "Gradient",
            Category::Optimization => "Optimization",
        }
        .into()
    }
}

#[derive(Clone, Debug)]
pub struct Instruction {
    opcode: OpCode,
    inputs: Rc<Vec<GenericTensor>>,
    outputs: Rc<Vec<GenericTensor>>,
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

#[macro_export]
macro_rules! inference_instruction {
    ( $opcode:expr, $inputs:expr, $outputs:expr, ) => {
        crate::instruction!($opcode, $inputs, $outputs, crate::Category::Inference,)
    };
}

#[macro_export]
macro_rules! loss_instruction {
    ( $opcode:expr, $inputs:expr, $outputs:expr, ) => {
        crate::instruction!($opcode, $inputs, $outputs, crate::Category::Loss,)
    };
}

#[macro_export]
macro_rules! gradient_instruction {
    ( $opcode:expr, $inputs:expr, $outputs:expr, ) => {
        crate::instruction!($opcode, $inputs, $outputs, crate::Category::Gradient,)
    };
}

#[macro_export]
macro_rules! optimization_instruction {
    ( $opcode:expr, $inputs:expr, $outputs:expr, ) => {
        crate::instruction!($opcode, $inputs, $outputs, crate::Category::Optimization,)
    };
}

impl Instruction {
    pub fn new(
        opcode: OpCode,
        inputs: &[&GenericTensor],
        outputs: &[&GenericTensor],
        category: Category,
        #[cfg(debug_assertions)] file: &str,
        #[cfg(debug_assertions)] line: u32,
        #[cfg(debug_assertions)] column: u32,
    ) -> Self {
        let inputs: Vec<GenericTensor> = inputs
            .to_owned()
            .into_iter()
            .map(|x| (*x).clone())
            .collect();
        let outputs: Vec<GenericTensor> = outputs
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
    pub fn inputs(&self) -> &Rc<Vec<GenericTensor>> {
        &self.inputs
    }
    pub fn outputs(&self) -> &Rc<Vec<GenericTensor>> {
        &self.outputs
    }
    pub fn execute(&self) -> Result<(), Error> {
        let inputs: Vec<&GenericTensor> = self.inputs.iter().collect();
        let outputs_f32: Vec<&GenericTensor> = self.outputs.iter().collect();
        self.opcode.execute(&inputs, &outputs_f32)
    }
}
