use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc, sync::Arc};

use crate::{tensor::Error, Instruction};

use super::{
    pipelines::{Controller, ExecutionUnit},
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
    let mut emit_pipeline = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut retire_pipeline = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut controller = Controller::new(streams, &emit_pipeline, &retire_pipeline);
    let mut execution_unit = ExecutionUnit::new(
        &emit_pipeline,
        &retire_pipeline,
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
    let mut emit_pipeline = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut retire_pipeline = Rc::new(RefCell::new(LinkedList::<usize>::new()));
    let mut controller = Controller::new(streams, &emit_pipeline, &retire_pipeline);
    let mut execution_unit = ExecutionUnit::new(
        &emit_pipeline,
        &retire_pipeline,
        &handler,
        streams,
        instructions,
    );
    controller.start();
    while execution_unit.step() || controller.step() {}
    handler.clone().deref().borrow().actual_transactions.clone()
}
