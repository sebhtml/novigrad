use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc, sync::Arc};

use crate::{tensor::Error, Instruction};

use super::{
    stream::Stream,
    transaction::{get_instruction_transactions, Transaction},
};

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
    simple_instructions: Rc<Vec<(Vec<usize>, Vec<usize>)>>,
    pub actual_transactions: Vec<Transaction>,
}

impl TransactionEmitter {
    pub fn new(
        _streams: &[Stream],
        simple_instructions: &Rc<Vec<(Vec<usize>, Vec<usize>)>>,
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

    fn spawn_stream(
        &mut self,
        stream: usize,
        streams: &[Stream],
        instructions: &Arc<Vec<Instruction>>,
    ) -> Result<(), Error> {
        let stream_instructions = streams[stream].instructions.clone();
        let instructions = instructions.clone();
        crate::execution_unit::ExecutionUnit::execute(stream_instructions, instructions)?;
        Ok(())
    }
}

impl StreamEventHandler for StreamExecutor {
    fn on_execute(
        &mut self,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
        stream: usize,
    ) -> Result<(), Error> {
        self.spawn_stream(stream, streams, instructions)
    }
}

#[allow(unused)]
pub fn execute_streams(
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    max_concurrent_streams: usize,
) {
    let mut handler = StreamExecutor::new();
    let handler = Rc::new(RefCell::new(handler));
    let mut fetch_queue = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut writeback_queue = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut controller = Scheduler::new(streams, &fetch_queue, &writeback_queue);
    let mut execution_unit = ExecutionUnit::new(
        &fetch_queue,
        &writeback_queue,
        &handler,
        streams,
        instructions,
    );
    controller.start();
    while execution_unit.step() || controller.step() {}
}

/// Simulate an execution of streams and emit operand transactions.
#[allow(unused)]
pub fn simulate_execution_and_collect_transactions(
    streams: &Arc<Vec<Stream>>,
    instructions: &Arc<Vec<Instruction>>,
    simple_instructions: &Rc<Vec<(Vec<usize>, Vec<usize>)>>,
    max_concurrent_streams: usize,
) -> Vec<Transaction> {
    let handler = TransactionEmitter::new(streams, simple_instructions);
    let handler = Rc::new(RefCell::new(handler));
    let mut fetch_queue = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut writeback_queue = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut controller = Scheduler::new(streams, &fetch_queue, &writeback_queue);
    let mut execution_unit = ExecutionUnit::new(
        &fetch_queue,
        &writeback_queue,
        &handler,
        streams,
        instructions,
    );
    controller.start();
    while execution_unit.step() || controller.step() {}
    handler.clone().deref().borrow().actual_transactions.clone()
}

pub struct Scheduler {
    dependents: Vec<Vec<usize>>,
    pending_dependencies: Vec<usize>,
    fetch_queue: Rc<RefCell<LinkedList<usize>>>,
    writeback_queue: Rc<RefCell<LinkedList<usize>>>,
}

impl Scheduler {
    pub fn new(
        streams: &[Stream],
        emit_queue: &Rc<RefCell<LinkedList<usize>>>,
        retire_queue: &Rc<RefCell<LinkedList<usize>>>,
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
            fetch_queue: emit_queue.clone(),
            writeback_queue: retire_queue.clone(),
        }
    }

    pub fn start(&self) {
        // Emit immediately all streams with no dependencies.
        for (stream, _) in self.pending_dependencies.iter().enumerate() {
            self.maybe_emit(stream);
        }
    }

    fn maybe_emit(&self, stream: usize) {
        let pending_dependencies = self.pending_dependencies[stream];
        if pending_dependencies == 0 {
            self.fetch_queue.deref().borrow_mut().push_back(stream);
        }
    }

    pub fn step(&mut self) -> bool {
        if let Some(stream) = self.writeback_queue.deref().borrow_mut().pop_front() {
            let dependents = &self.dependents[stream];
            for dependent in dependents.iter() {
                self.pending_dependencies[*dependent] -= 1;
                self.maybe_emit(*dependent);
            }
            true
        } else {
            false
        }
    }
}

/// https://en.wikipedia.org/wiki/Instruction_pipelining
pub struct ExecutionUnit<Handler: StreamEventHandler> {
    handler: Rc<RefCell<Handler>>,
    streams: Arc<Vec<Stream>>,
    instructions: Arc<Vec<Instruction>>,
    fetch_queue: Rc<RefCell<LinkedList<usize>>>,
    writeback_queue: Rc<RefCell<LinkedList<usize>>>,
}

impl<Handler: StreamEventHandler + Clone> ExecutionUnit<Handler> {
    pub fn new(
        fetch_queue: &Rc<RefCell<LinkedList<usize>>>,
        writeback_queue: &Rc<RefCell<LinkedList<usize>>>,
        handler: &Rc<RefCell<Handler>>,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        Self {
            handler: handler.clone(),
            streams: streams.clone(),
            instructions: instructions.clone(),
            fetch_queue: fetch_queue.clone(),
            writeback_queue: writeback_queue.clone(),
        }
    }

    pub fn step(&mut self) -> bool {
        // Fetch
        if let Some(stream) = self.fetch_queue.deref().borrow_mut().pop_front() {
            // Call handler to execute the instructions for that stream.
            self.handler
                .deref()
                .borrow_mut()
                .on_execute(&self.streams, &self.instructions, stream)
                .unwrap();
            // Writeback
            self.writeback_queue.deref().borrow_mut().push_back(stream);
            true
        } else {
            false
        }
    }
}
