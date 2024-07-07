use crate::{
    opcode::OpCode,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, OperatorAttributes,
};
use std::{ops::Deref, sync::Arc};

#[derive(Clone, Debug, PartialEq)]
pub enum Category {
    EnableDropout,
    DisableDropout,
    Inference,
    Loss,
    Gradient,
    Optimization,
}

pub fn is_forward_category(category: &Category) -> bool {
    [
        Category::EnableDropout,
        Category::DisableDropout,
        Category::Inference,
        Category::Loss,
    ]
    .contains(category)
}

impl From<Category> for String {
    fn from(val: Category) -> Self {
        match val {
            Category::EnableDropout => "EnableDropout",
            Category::DisableDropout => "DisableDropout",
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
    attributes: OperatorAttributes,
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
    ( $opcode:expr, $attributes:expr, $inputs:expr, $outputs:expr, $category:expr, ) => {
        $crate::Instruction::new(
            $opcode,
            $attributes,
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
    ( $opcode:expr, $attributes:expr, $inputs:expr, $outputs:expr, $category:expr, ) => {
        crate::Instruction::new($opcode, $attributes, $inputs, $outputs, $category)
    };
}

impl Instruction {
    pub fn new(
        opcode: OpCode,
        attributes: OperatorAttributes,
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
            attributes,
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
    pub fn execute(&self, device: &Device, device_stream: &DeviceStream) -> Result<(), Error> {
        let attributes = &self.attributes;
        let inputs: Vec<&Tensor> = self.inputs.iter().collect();
        #[cfg(debug_assertions)]
        {
            for input in inputs.iter() {
                debug_assert_eq!(false, input.is_nan()?, "{:?}", self);
                debug_assert_eq!(false, input.is_infinite()?, "{:?}", self);
            }
        }
        let outputs: Vec<&Tensor> = self.outputs.iter().collect();
        self.opcode
            .execute(attributes, &inputs, &outputs, device, device_stream)?;
        #[cfg(debug_assertions)]
        {
            for output in outputs.iter() {
                debug_assert_eq!(false, output.is_nan()?, "{:?}", self);
                debug_assert_eq!(false, output.is_infinite()?, "{:?}", self);
            }
        }
        Ok(())
    }
}

pub fn filter_instructions(
    instructions: Vec<Instruction>,
    filter: Option<Category>,
) -> Vec<Instruction> {
    match filter {
        Some(category) => instructions
            .into_iter()
            .filter(|x| x.category() == category)
            .collect(),
        None => instructions,
    }
}
