use std::sync::{Arc, Mutex};
pub mod cpu_scheduler;
pub mod transaction;

use crate::{stream::DeviceStream, streams::stream::Stream, tensor::Error, Device, Instruction};

pub trait SchedulerTrait<Handler>
where
    Handler: StreamEventHandler,
{
    fn new(
        device: &Device,
        execution_units_len: usize,
        streams: &Arc<Vec<Stream>>,
        handler: &Handler,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self;

    fn start(&mut self);

    fn stop(&mut self);

    fn execute(&mut self);
}

pub trait StreamEventHandler {
    fn on_execute(
        &mut self,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
        stream: usize,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;
}

#[derive(Clone)]
pub struct InstructionEmitter {
    pub executed_instructions: Arc<Mutex<Vec<usize>>>,
}

impl InstructionEmitter {
    pub fn new() -> Self {
        Self {
            executed_instructions: Default::default(),
        }
    }
}

impl StreamEventHandler for InstructionEmitter {
    fn on_execute(
        &mut self,
        streams: &Arc<Vec<Stream>>,
        _instructions: &Arc<Vec<Instruction>>,
        stream: usize,
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let stream_instructions = &streams[stream].instructions;
        for instruction in stream_instructions.iter() {
            self.executed_instructions
                .lock()
                .unwrap()
                .push(*instruction);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct StreamExecutor {}

impl StreamExecutor {
    pub fn new() -> Self {
        Self {}
    }
}

impl StreamEventHandler for StreamExecutor {
    fn on_execute(
        &mut self,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
        stream: usize,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let stream_instructions = streams[stream].instructions.clone();
        let instructions = instructions.clone();
        for i in stream_instructions.iter() {
            let instruction = &instructions[*i];
            instruction.execute(device_stream)?;
        }
        Ok(())
    }
}