use crate::{tensor::Error, Instruction};

pub struct ExecutionUnit {}

impl ExecutionUnit {
    pub fn execute(
        stream_instructions: &[usize],
        instructions: &[Instruction],
    ) -> Result<(), Error> {
        for i in stream_instructions.iter() {
            let instruction = &instructions[*i];
            instruction.execute()?;
        }
        Ok(())
    }
}
