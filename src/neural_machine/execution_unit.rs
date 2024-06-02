use std::ops::Deref;

use crate::{tensor::Error, Instruction};

pub struct ExecutionUnit {}

impl ExecutionUnit {
    pub fn execute(
        stream_instructions: impl Deref<Target = Vec<usize>>,
        instructions: impl Deref<Target = Vec<Instruction>>,
    ) -> Result<(), Error> {
        for i in stream_instructions.iter() {
            let instruction = &instructions[*i];
            instruction.execute()?;
        }
        Ok(())
    }
}
