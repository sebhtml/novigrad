use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    OpCode,
};
use std::{ops::Deref, sync::Arc};

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
    inputs: Arc<Vec<Tensor>>,
    outputs: Arc<Vec<Tensor>>,
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
        crate::Instruction::new(
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
        crate::Instruction::new($opcode, $inputs, $outputs, $category)
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
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        category: Category,
        #[cfg(debug_assertions)] file: &str,
        #[cfg(debug_assertions)] line: u32,
        #[cfg(debug_assertions)] column: u32,
    ) -> Self {
        let inputs: Vec<Tensor> = inputs
            .to_owned()
            .into_iter()
            .map(|x| (*x).clone())
            .collect();
        let outputs: Vec<Tensor> = outputs
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
    pub fn inputs(&self) -> impl Deref<Target = Vec<Tensor>> + '_ {
        self.inputs.deref()
    }
    pub fn outputs(&self) -> impl Deref<Target = Vec<Tensor>> + '_ {
        self.outputs.deref()
    }
    pub fn execute(&self, device_stream: &DeviceStream) -> Result<(), Error> {
        let inputs: Vec<&Tensor> = self.inputs.iter().collect();
        let outputs: Vec<&Tensor> = self.outputs.iter().collect();
        self.opcode.execute(&inputs, &outputs, device_stream)
    }
}

pub fn filter_instructions(
    instructions: Vec<Instruction>,
    filter: Option<Category>,
) -> Vec<Instruction> {
    let instructions = match filter {
        Some(category) => instructions
            .into_iter()
            .filter(|x| x.category() == category)
            .collect(),
        None => instructions,
    };
    instructions
}
