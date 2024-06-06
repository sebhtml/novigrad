use std::{collections::BTreeSet, ops::Deref, thread::JoinHandle};

use crate::{execution_unit::ExecutionUnit, tensor::Error, Instruction};

use super::{
    stream::Stream,
    transaction::{get_instruction_transactions, Transaction},
};

pub trait StreamEventHandler {
    fn new(streams: &[Stream]) -> Self;
    fn on_spawn(
        &mut self,
        streams: &[Stream],
        instructions: &[Instruction],
        simple_instructions: &[(Vec<usize>, Vec<usize>)],
        stream: usize,
    );
    fn on_join(
        &mut self,
        streams: &[Stream],
        instructions: &[Instruction],
        simple_instructions: &[(Vec<usize>, Vec<usize>)],
        stream: usize,
    );
}

pub struct TransactionEmitter {
    pub actual_transactions: Vec<Transaction>,
}

impl StreamEventHandler for TransactionEmitter {
    fn new(_streams: &[Stream]) -> Self {
        Self {
            actual_transactions: Default::default(),
        }
    }
    fn on_spawn(
        &mut self,
        _streams: &[Stream],
        _instructions: &[Instruction],
        _simple_instructions: &[(Vec<usize>, Vec<usize>)],
        _stream: usize,
    ) {
    }
    fn on_join(
        &mut self,
        streams: &[Stream],
        _instructions: &[Instruction],
        simple_instructions: &[(Vec<usize>, Vec<usize>)],
        stream: usize,
    ) {
        let stream_instructions = &streams[stream].instructions;

        for instruction in stream_instructions.iter() {
            let instruction = *instruction;
            let (inputs, outputs) = &simple_instructions[instruction];
            let mut instruction_transactions =
                get_instruction_transactions(instruction, inputs, outputs);
            self.actual_transactions
                .extend_from_slice(&mut instruction_transactions);
        }
    }
}

pub struct StreamExecutor {
    threads: Vec<Option<JoinHandle<Result<(), Error>>>>,
}

impl StreamEventHandler for StreamExecutor {
    fn new(streams: &[Stream]) -> Self {
        let mut threads: Vec<Option<JoinHandle<Result<(), Error>>>> = vec![];
        for _ in 0..streams.len() {
            threads.push(None);
        }
        Self { threads }
    }
    fn on_spawn(
        &mut self,
        streams: &[Stream],
        instructions: &[Instruction],
        _simple_instructions: &[(Vec<usize>, Vec<usize>)],
        stream: usize,
    ) {
        spawn_stream(stream, streams, &mut self.threads, instructions).unwrap();
    }

    fn on_join(
        &mut self,
        streams: &[Stream],
        _instructions: &[Instruction],
        _simple_instructions: &[(Vec<usize>, Vec<usize>)],
        stream: usize,
    ) {
        join_stream(stream, streams, &mut self.threads).unwrap();
    }
}

#[allow(unused)]
pub fn execute_streams(
    streams: &[Stream],
    instructions: &[Instruction],
    simple_instructions: &[(Vec<usize>, Vec<usize>)],
    max_concurrent_streams: usize,
) {
    let mut handler = StreamExecutor::new(streams);
    execute_streams_v2(
        streams,
        instructions,
        simple_instructions,
        max_concurrent_streams,
        &mut handler,
    );
}

/// Simulate an execution of streams and emit operand transactions.
#[allow(unused)]
pub fn simulate_execution_and_collect_transactions(
    streams: &[Stream],
    instructions: &[Instruction],
    simple_instructions: &[(Vec<usize>, Vec<usize>)],
    max_concurrent_streams: usize,
) -> Vec<Transaction> {
    let mut handler = TransactionEmitter::new(streams);
    execute_streams_v2(
        streams,
        instructions,
        simple_instructions,
        max_concurrent_streams,
        &mut handler,
    );
    handler.actual_transactions
}

pub fn execute_streams_v2(
    streams: &[Stream],
    instructions: &[Instruction],
    simple_instructions: &[(Vec<usize>, Vec<usize>)],
    max_concurrent_streams: usize,
    handler: &mut impl StreamEventHandler,
) {
    let mut unreached_streams = BTreeSet::<usize>::new();
    for i in 0..streams.len() {
        unreached_streams.insert(i);
    }
    let mut spawned_streams = BTreeSet::<usize>::new();
    let mut joined_streams = BTreeSet::<usize>::new();

    while joined_streams.len() != streams.len() {
        let mut stream_to_spawn: Option<usize> = None;
        // Find a stream that can be spawned.
        for unreached_stream in unreached_streams.iter() {
            let mut can_spawn = true;
            let dependencies = &streams[*unreached_stream].dependencies;
            for dependency in dependencies {
                if !joined_streams.contains(dependency) {
                    can_spawn = false;
                    break;
                }
            }
            if can_spawn {
                stream_to_spawn = Some(*unreached_stream);
                break;
            }
        }
        if let Some(stream_to_spawn) = stream_to_spawn {
            let concurrent_streams = spawned_streams.len() - joined_streams.len();
            if concurrent_streams == max_concurrent_streams {
                // Join the oldest active stream before spawning this one.
                let oldest = spawned_streams.iter().min().map(|x| *x);
                if let Some(oldest) = oldest {
                    joined_streams.insert(oldest);
                }
            }
            // Spawn it.
            unreached_streams.remove(&stream_to_spawn);
            spawned_streams.insert(stream_to_spawn);
            // Emit transactions on the execution unit pipeline.
            handler.on_spawn(streams, instructions, simple_instructions, stream_to_spawn);
            // Immediately join the thread.
            joined_streams.insert(stream_to_spawn);
            handler.on_join(streams, instructions, simple_instructions, stream_to_spawn);
        }
    }
}

fn join_stream(
    _stream: usize,
    _streams: &[Stream],
    _threads: &mut Vec<Option<JoinHandle<Result<(), Error>>>>,
) -> Result<(), Error> {
    /*
    let thread = threads[stream].take();
    if let Some(thread) = thread {
        match thread.join() {
            Ok(result) => match result {
                Ok(_) => (),
                Err(err) => return Err(err),
            },
            Err(_) => return Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }
     */
    #[cfg(feature = "verbose_streams")]
    println!(
        "Transition stream {}  {} -> {}",
        stream, streams[stream].state, new_state
    );
    #[cfg(feature = "verbose_streams")]
    println!("active_streams {}", active_streams.len());
    Ok(())
}

fn spawn_stream(
    stream: usize,
    streams: &[Stream],
    _threads: &mut Vec<Option<JoinHandle<Result<(), Error>>>>,
    instructions: &[Instruction],
) -> Result<(), Error> {
    let stream_instructions: &[usize] = streams[stream].instructions.deref().deref();
    ExecutionUnit::execute(stream_instructions, instructions)?;

    /*
    let spawned_thread =
        thread::spawn(|| ExecutionUnit::execute(stream_instructions, instructions));
    threads[stream] = Some(spawned_thread);
    */
    Ok(())
}
