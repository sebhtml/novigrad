#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};

mod controller;
mod execution_unit;
pub mod scheduler;
use scheduler::Scheduler;
use transaction::{get_instruction_transactions, Transaction};

use crate::{
    schedulers::SchedulerTrait, stream::DeviceStream, streams::stream::Stream, tensor::Error,
    Device, Instruction,
};
pub mod queue;
pub mod transaction;

pub enum Command {
    Execute,
    Stop,
    WorkUnitDispatch(usize),
    WorkUnitCompletion(usize),
    ExecutionCompletion,
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
pub struct TransactionEmitter {
    simple_instructions: Arc<Vec<(Vec<usize>, Vec<usize>)>>,
    pub actual_transactions: Arc<Mutex<Vec<Transaction>>>,
}

impl TransactionEmitter {
    pub fn new(simple_instructions: &Arc<Vec<(Vec<usize>, Vec<usize>)>>) -> Self {
        Self {
            simple_instructions: simple_instructions.clone(),
            actual_transactions: Default::default(),
        }
    }
}

impl StreamEventHandler for TransactionEmitter {
    fn on_execute(
        &mut self,
        streams: &Arc<Vec<Stream>>,
        _instructions: &Arc<Vec<Instruction>>,
        stream: usize,
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let stream_instructions = &streams[stream].instructions;
        for instruction in stream_instructions.iter() {
            let instruction = *instruction;
            let (inputs, outputs) = &self.simple_instructions[instruction];
            let mut instruction_transactions =
                get_instruction_transactions(instruction, inputs, outputs);
            self.actual_transactions
                .lock()
                .unwrap()
                .extend_from_slice(&mut instruction_transactions);
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

#[allow(unused)]
pub fn execute_streams(
    device: &Device,
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    execution_units_len: usize,
) {
    let mut handler = StreamExecutor::new();
    run_scheduler(device, streams, instructions, execution_units_len, &handler);
}

/// Simulate an execution of streams and emit operand transactions.
#[allow(unused)]
pub fn simulate_execution_and_collect_transactions(
    device: &Device,
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    simple_instructions: &Arc<Vec<(Vec<usize>, Vec<usize>)>>,
    execution_units_len: usize,
) -> Vec<Transaction> {
    let handler = TransactionEmitter::new(simple_instructions);
    run_scheduler(device, streams, instructions, execution_units_len, &handler);
    handler.clone().actual_transactions.lock().unwrap().clone()
}

/// Simulate an execution of streams and emit executed instructions.
#[allow(unused)]
pub fn simulate_execution_and_collect_instructions(
    device: &Device,
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    execution_units_len: usize,
) -> Vec<usize> {
    let handler = InstructionEmitter::new();
    run_scheduler(device, streams, instructions, execution_units_len, &handler);
    handler
        .clone()
        .executed_instructions
        .lock()
        .unwrap()
        .clone()
}

pub fn run_scheduler<Handler>(
    device: &Device,
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    execution_units_len: usize,
    handler: &Handler,
) where
    Handler: StreamEventHandler + Clone + Send + Sync + 'static,
{
    let mut scheduler = Scheduler::new(device, execution_units_len, streams, handler, instructions);
    scheduler.start();
    scheduler.execute();
    scheduler.stop();
}
