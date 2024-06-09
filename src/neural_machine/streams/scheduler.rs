use std::{
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use crate::{tensor::Error, Instruction};

use super::{
    queue::Queue,
    stream::Stream,
    transaction::{get_instruction_transactions, Transaction},
};

const STOP: usize = usize::MAX;

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

fn run_scheduler<Handler: StreamEventHandler + Clone + Send + Sync + 'static>(
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    max_concurrent_streams: usize,
    handler: &Handler,
) {
    let mut scheduler = Scheduler::new(max_concurrent_streams, streams, handler, instructions);
    scheduler.start();
    scheduler.stop();
}

pub struct Controller {
    dependents: Vec<Vec<usize>>,
    pending_dependencies: Vec<usize>,
    dispatch_queues: Vec<Arc<Queue<usize>>>,
    completion_queue: Arc<Queue<usize>>,
    completed_streams: usize,
    max_concurrent_streams: usize,
}

impl Controller {
    pub fn new(
        streams: &[Stream],
        dispatch_queues: &Vec<Arc<Queue<usize>>>,
        completion_queue: &Arc<Queue<usize>>,
        max_concurrent_streams: usize,
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
            pending_dependencies,
            dispatch_queues: dispatch_queues.clone(),
            completion_queue: completion_queue.clone(),
            completed_streams: 0,
            max_concurrent_streams,
        }
    }

    pub fn spawn(mut controller: Self) -> JoinHandle<Self> {
        // Dispatch immediately all streams with no dependencies.
        for (stream, _) in controller.pending_dependencies.iter().enumerate() {
            controller.maybe_dispatch(stream);
        }

        let handle = thread::spawn(|| {
            while controller.step() {}
            controller
        });
        handle
    }

    fn maybe_dispatch(&self, stream: usize) {
        let pending_dependencies = self.pending_dependencies[stream];
        if pending_dependencies == 0 {
            let ordinal = stream % self.max_concurrent_streams;
            self.dispatch_queues[ordinal].push_back(stream);
        }
    }

    pub fn step(&mut self) -> bool {
        let stream = self.completion_queue.pop_front();
        if let Some(stream) = stream {
            self.completed_streams += 1;
            let dependents = &self.dependents[stream];
            for dependent in dependents.iter() {
                self.pending_dependencies[*dependent] -= 1;
                self.maybe_dispatch(*dependent);
            }
        }

        if self.completed_streams == self.dependents.len() {
            for ordinal in 0..self.max_concurrent_streams {
                self.dispatch_queues[ordinal].push_back(STOP);
            }
            false
        } else {
            true
        }
    }
}

/// https://en.wikipedia.org/wiki/Instruction_pipelining
pub struct ExecutionUnit<Handler: StreamEventHandler + Send + Sync> {
    #[allow(unused)]
    ordinal: usize,
    handler: Handler,
    streams: Arc<Vec<Stream>>,
    instructions: Arc<Vec<Instruction>>,
    dispatch_queue: Arc<Queue<usize>>,
    completion_queue: Arc<Queue<usize>>,
    completed_items: usize,
}

impl<Handler: StreamEventHandler + Send + Sync + 'static> ExecutionUnit<Handler> {
    pub fn new(
        ordinal: usize,
        dispatch_queue: &Arc<Queue<usize>>,
        completion_queue: &Arc<Queue<usize>>,
        handler: Handler,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        Self {
            ordinal,
            handler,
            streams: streams.clone(),
            instructions: instructions.clone(),
            dispatch_queue: dispatch_queue.clone(),
            completion_queue: completion_queue.clone(),
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
        let stream = self.dispatch_queue.pop_front();
        if let Some(stream) = stream {
            if stream == STOP {
                return false;
            }
            //println!("execution unit ordinal {} dispatch {:?}", self.ordinal, Instant::now());
            // Call handler to execute the instructions for that stream.
            self.handler
                .on_execute(&self.streams, &self.instructions, stream)
                .unwrap();
            //println!("execution unit ordinal {} completion {:?}", self.ordinal, Instant::now());
            // Writeback
            self.completion_queue.push_back(stream);
            self.completed_items += 1;
        }
        true
    }
}

impl<Handler: StreamEventHandler + Send + Sync> Drop for ExecutionUnit<Handler> {
    fn drop(&mut self) {
        //println!("execution unit: {}, completed_items: {}", self.ordinal, self.completed_items);
    }
}

pub struct Scheduler<Handler: StreamEventHandler + Send + Sync> {
    controller: Option<Controller>,
    execution_units: Option<Vec<ExecutionUnit<Handler>>>,
    controller_handle: Option<JoinHandle<Controller>>,
    execution_unit_handles: Option<Vec<JoinHandle<ExecutionUnit<Handler>>>>,
}

impl<Handler: StreamEventHandler + Clone + Send + Sync + 'static> Scheduler<Handler> {
    pub fn new(
        max_concurrent_streams: usize,
        streams: &Arc<Vec<Stream>>,
        handler: &Handler,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        // Create structures
        let dispatch_queues = (0..max_concurrent_streams)
            .map(|_| Arc::new(Queue::<usize>::default()))
            .collect::<Vec<_>>();
        let completion_queue = Arc::new(Queue::default());
        let controller = Controller::new(
            streams,
            &dispatch_queues,
            &completion_queue,
            max_concurrent_streams,
        );
        let execution_units = (0..max_concurrent_streams)
            .map(|ordinal| {
                let execution_unit = ExecutionUnit::new(
                    ordinal,
                    &dispatch_queues[ordinal],
                    &completion_queue,
                    handler.clone(),
                    streams,
                    instructions,
                );
                execution_unit
            })
            .collect::<Vec<_>>();
        Self {
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
    pub fn start(&mut self) {
        // Spawn threads
        let execution_unit_handles = self
            .execution_units
            .take()
            .unwrap()
            .into_iter()
            .map(|execution_unit| ExecutionUnit::spawn(execution_unit))
            .collect::<Vec<_>>();
        let controller_handle = Controller::spawn(self.controller.take().unwrap());
        self.execution_unit_handles = Some(execution_unit_handles);
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
    pub fn stop(&mut self) {
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
}
