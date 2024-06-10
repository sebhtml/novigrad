#[cfg(test)]
mod tests;

use std::{
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use queue::Queue;

use crate::{tensor::Error, Instruction};
pub mod queue;
use super::streams::{
    stream::Stream,
    transaction::{get_instruction_transactions, Transaction},
};

pub enum Command {
    Execute,
    Stop,
    Dispatch(usize),
    Completion(usize),
}

pub trait StreamEventHandler {
    fn on_execute(
        &mut self,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
        stream: usize,
    ) -> Result<(), Error>;
}

#[derive(Clone)]
pub struct TransactionEmitter {
    simple_instructions: Arc<Vec<(Vec<usize>, Vec<usize>)>>,
    pub actual_transactions: Arc<Mutex<Vec<Transaction>>>,
}

impl TransactionEmitter {
    pub fn new(
        _streams: &[Stream],
        simple_instructions: &Arc<Vec<(Vec<usize>, Vec<usize>)>>,
    ) -> Self {
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
    ) -> Result<(), Error> {
        let stream_instructions = streams[stream].instructions.clone();
        let instructions = instructions.clone();
        crate::execution_unit::ExecutionUnit::execute(stream_instructions, instructions)?;
        Ok(())
    }
}

#[allow(unused)]
pub fn execute_streams(
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    max_concurrent_streams: usize,
) {
    let mut handler = StreamExecutor::new();
    run_scheduler(streams, instructions, max_concurrent_streams, &handler);
}

/// Simulate an execution of streams and emit operand transactions.
#[allow(unused)]
pub fn simulate_execution_and_collect_transactions(
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    simple_instructions: &Arc<Vec<(Vec<usize>, Vec<usize>)>>,
    max_concurrent_streams: usize,
) -> Vec<Transaction> {
    let handler = TransactionEmitter::new(streams, simple_instructions);
    run_scheduler(streams, instructions, max_concurrent_streams, &handler);
    handler.clone().actual_transactions.lock().unwrap().clone()
}

fn run_scheduler<Handler>(
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    max_concurrent_streams: usize,
    handler: &Handler,
) where
    Handler: StreamEventHandler + Clone + Send + Sync + 'static,
{
    let mut scheduler = Scheduler::new(max_concurrent_streams, streams, handler, instructions);
    scheduler.spawn();
    scheduler.execute();
    scheduler.join();
}

pub struct Controller {
    dependents: Vec<Vec<usize>>,
    initial_pending_dependencies: Vec<usize>,
    current_pending_dependencies: Vec<usize>,
    execution_unit_command_queues: Vec<Arc<Queue<Command>>>,
    controller_command_queue: Arc<Queue<Command>>,
    completed_streams: usize,
    max_concurrent_streams: usize,
}

impl Controller {
    pub fn new(
        streams: &[Stream],
        execution_unit_command_queues: &Vec<Arc<Queue<Command>>>,
        controller_command_queue: &Arc<Queue<Command>>,
        execution_units_len: usize,
    ) -> Self {
        let pending_dependencies = streams.iter().map(|x| x.dependencies.len()).collect();
        let mut dependents = vec![vec![]; streams.len()];
        for (dependent, dependencies) in streams.iter().map(|x| &x.dependencies).enumerate() {
            for dependency in dependencies.iter() {
                dependents[*dependency].push(dependent);
            }
        }
        Self {
            dependents,
            initial_pending_dependencies: pending_dependencies,
            current_pending_dependencies: Default::default(),
            execution_unit_command_queues: execution_unit_command_queues.clone(),
            controller_command_queue: controller_command_queue.clone(),
            completed_streams: 0,
            max_concurrent_streams: execution_units_len,
        }
    }

    pub fn spawn(mut controller: Self) -> JoinHandle<Self> {
        let handle = thread::spawn(|| {
            while controller.step() {}
            controller
        });
        handle
    }

    fn maybe_dispatch(&self, stream: usize) {
        let pending_dependencies = self.current_pending_dependencies[stream];
        if pending_dependencies == 0 {
            let ordinal = stream % self.max_concurrent_streams;
            self.execution_unit_command_queues[ordinal].push_back(Command::Dispatch(stream));
        }
    }

    pub fn step(&mut self) -> bool {
        let command = self.controller_command_queue.pop_front();
        match command {
            Some(Command::Execute) => {
                self.current_pending_dependencies = self.initial_pending_dependencies.clone();
                // Dispatch immediately all streams with no dependencies.
                for (stream, _) in self.current_pending_dependencies.iter().enumerate() {
                    self.maybe_dispatch(stream);
                }
            }
            Some(Command::Completion(stream)) => {
                self.completed_streams += 1;
                let dependents = &self.dependents[stream];
                for dependent in dependents.iter() {
                    self.current_pending_dependencies[*dependent] -= 1;
                    self.maybe_dispatch(*dependent);
                }
            }
            _ => {}
        }

        if self.completed_streams == self.dependents.len() {
            for ordinal in 0..self.max_concurrent_streams {
                self.execution_unit_command_queues[ordinal].push_back(Command::Stop);
            }
            false
        } else {
            true
        }
    }
}

/// https://en.wikipedia.org/wiki/Instruction_pipelining
pub struct ExecutionUnit<Handler>
where
    Handler: StreamEventHandler + Send + Sync,
{
    #[allow(unused)]
    ordinal: usize,
    handler: Handler,
    streams: Arc<Vec<Stream>>,
    instructions: Arc<Vec<Instruction>>,
    execution_unit_command_queue: Arc<Queue<Command>>,
    controller_command_queue: Arc<Queue<Command>>,
    completed_items: usize,
}

impl<Handler> ExecutionUnit<Handler>
where
    Handler: StreamEventHandler + Send + Sync + 'static,
{
    pub fn new(
        ordinal: usize,
        execution_unit_command_queue: &Arc<Queue<Command>>,
        controller_command_queue: &Arc<Queue<Command>>,
        handler: Handler,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        Self {
            ordinal,
            handler,
            streams: streams.clone(),
            instructions: instructions.clone(),
            execution_unit_command_queue: execution_unit_command_queue.clone(),
            controller_command_queue: controller_command_queue.clone(),
            completed_items: 0,
        }
    }

    pub fn spawn(mut execution_unit: Self) -> JoinHandle<Self> {
        let handle = thread::spawn(|| {
            while execution_unit.step() {}
            execution_unit
        });
        handle
    }

    fn step(&mut self) -> bool {
        // Fetch
        let command = self.execution_unit_command_queue.pop_front();
        match command {
            Some(Command::Stop) => {
                return false;
            }
            Some(Command::Dispatch(stream)) => {
                // Call handler to execute the instructions for that stream.
                self.handler
                    .on_execute(&self.streams, &self.instructions, stream)
                    .unwrap();
                // Writeback
                self.controller_command_queue
                    .push_back(Command::Completion(stream));
                self.completed_items += 1;
            }
            _ => {}
        }
        true
    }
}

pub struct Scheduler<Handler>
where
    Handler: StreamEventHandler + Send + Sync,
{
    controller_command_queue: Arc<Queue<Command>>,
    controller: Option<Controller>,
    execution_units: Option<Vec<ExecutionUnit<Handler>>>,
    controller_handle: Option<JoinHandle<Controller>>,
    execution_unit_handles: Option<Vec<JoinHandle<ExecutionUnit<Handler>>>>,
}

impl<Handler> Scheduler<Handler>
where
    Handler: StreamEventHandler + Clone + Send + Sync + 'static,
{
    pub fn new(
        max_concurrent_streams: usize,
        streams: &Arc<Vec<Stream>>,
        handler: &Handler,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        // Create structures
        let execution_unit_command_queues = (0..max_concurrent_streams)
            .map(|_| Arc::new(Queue::<Command>::default()))
            .collect::<Vec<_>>();
        let controller_command_queue = Arc::new(Queue::default());
        let controller = Controller::new(
            streams,
            &execution_unit_command_queues,
            &controller_command_queue,
            max_concurrent_streams,
        );
        let execution_units = (0..max_concurrent_streams)
            .map(|ordinal| {
                let execution_unit = ExecutionUnit::new(
                    ordinal,
                    &execution_unit_command_queues[ordinal],
                    &controller_command_queue,
                    handler.clone(),
                    streams,
                    instructions,
                );
                execution_unit
            })
            .collect::<Vec<_>>();
        Self {
            controller_command_queue,
            controller: Some(controller),
            execution_units: Some(execution_units),
            controller_handle: None,
            execution_unit_handles: None,
        }
    }

    /// Pre-conditions
    /// - self.execution_units is some
    /// - self.execution_unit_handles is none
    /// - self.controller is some
    /// - self.controller_handle is none
    /// Post-conditions
    /// - self.execution_units is none
    /// - self.execution_unit_handles is some
    /// - self.controller is none
    /// - self.controller_handle is some
    pub fn spawn(&mut self) {
        // Spawn threads
        let execution_unit_handles = self
            .execution_units
            .take()
            .unwrap()
            .into_iter()
            .map(|execution_unit| ExecutionUnit::spawn(execution_unit))
            .collect::<Vec<_>>();
        self.execution_unit_handles = Some(execution_unit_handles);

        let controller_handle = Controller::spawn(self.controller.take().unwrap());
        self.controller_handle = Some(controller_handle);
    }

    /// Pre-conditions
    /// - self.execution_units is none
    /// - self.execution_unit_handles is some
    /// - self.controller is none
    /// - self.controller_handle is some
    /// Post-conditions
    /// - self.execution_units is some
    /// - self.execution_unit_handles is none
    /// - self.controller is some
    /// - self.controller_handle is none
    pub fn join(&mut self) {
        let controller = self.controller_handle.take().unwrap().join().unwrap();
        let execution_units = self
            .execution_unit_handles
            .take()
            .unwrap()
            .into_iter()
            .map(|x| x.join().unwrap())
            .collect::<Vec<_>>();
        self.controller = Some(controller);
        self.execution_units = Some(execution_units);
    }

    pub fn execute(&mut self) {
        self.controller_command_queue.push_back(Command::Execute);
    }
}
