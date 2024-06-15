#[cfg(test)]
mod tests;

use std::sync::Arc;

mod controller;
mod execution_unit;
pub mod scheduler;
use scheduler::CpuStreamScheduler;
use transaction::Transaction;

use crate::{
    schedulers::{
        InstructionEmitter, SchedulerTrait, StreamEventHandler, StreamExecutor, TransactionEmitter,
    },
    streams::stream::Stream,
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
    let mut scheduler = CpuStreamScheduler::new(device, execution_units_len, streams, handler, instructions);
    scheduler.start();
    scheduler.execute();
    scheduler.stop();
}
