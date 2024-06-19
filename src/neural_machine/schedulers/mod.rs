use std::sync::{Arc, Mutex};
pub mod cpu_scheduler;
pub mod gpu_scheduler;
pub mod transaction;
pub mod verification;

#[allow(unused)]
use cpu_scheduler::scheduler::CpuStreamScheduler;
#[allow(unused)]
use gpu_scheduler::GpuStreamScheduler;
use transaction::{Transaction, TransactionEmitter};

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
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;
}

#[derive(Clone)]
pub struct InstructionEmitter {
    pub executed_instructions: Arc<Mutex<Vec<usize>>>,
}

impl Default for InstructionEmitter {
    fn default() -> Self {
        Self::new()
    }
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
        _device: &Device,
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

impl Default for StreamExecutor {
    fn default() -> Self {
        Self::new()
    }
}

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
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let stream_instructions = streams[stream].instructions.clone();
        let instructions = instructions.clone();
        for i in stream_instructions.iter() {
            let instruction = &instructions[*i];
            instruction.execute(device, device_stream)?;
        }
        Ok(())
    }
}

#[allow(unused)]
pub fn execute_streams<Scheduler>(
    device: &Device,
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    execution_units_len: usize,
) where
    Scheduler: SchedulerTrait<StreamExecutor>,
{
    let mut handler = StreamExecutor::new();
    let mut scheduler =
        Scheduler::new(device, execution_units_len, streams, &handler, instructions);
    run_scheduler(&mut scheduler);
}

/// Simulate an execution of streams and emit operand transactions.
#[allow(unused)]
pub fn simulate_execution_and_collect_transactions<Scheduler>(
    device: &Device,
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    simple_instructions: &Arc<Vec<(Vec<usize>, Vec<usize>)>>,
    execution_units_len: usize,
) -> Vec<Transaction>
where
    Scheduler: SchedulerTrait<TransactionEmitter>,
{
    let handler = TransactionEmitter::new(simple_instructions);
    let mut scheduler =
        Scheduler::new(device, execution_units_len, streams, &handler, instructions);
    run_scheduler(&mut scheduler);
    handler.clone().actual_transactions.lock().unwrap().clone()
}

/// Simulate an execution of streams and emit executed instructions.
#[allow(unused)]
pub fn simulate_execution_and_collect_instructions<Scheduler>(
    device: &Device,
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    execution_units_len: usize,
) -> Vec<usize>
where
    Scheduler: SchedulerTrait<InstructionEmitter>,
{
    let handler = InstructionEmitter::new();
    let mut scheduler =
        Scheduler::new(device, execution_units_len, streams, &handler, instructions);
    run_scheduler(&mut scheduler);
    handler
        .clone()
        .executed_instructions
        .lock()
        .unwrap()
        .clone()
}

pub fn run_scheduler<Handler>(scheduler: &mut impl SchedulerTrait<Handler>)
where
    Handler: StreamEventHandler + Clone + Send + Sync + 'static,
{
    scheduler.start();
    scheduler.execute();
    scheduler.stop();
}

#[cfg(feature = "cuda")]
pub type DefaultStreamScheduler = GpuStreamScheduler<StreamExecutor>;
#[cfg(not(feature = "cuda"))]
pub type DefaultStreamScheduler = CpuStreamScheduler<StreamExecutor>;
