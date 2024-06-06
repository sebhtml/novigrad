use std::{collections::BTreeSet, sync::Arc, thread::JoinHandle};

use crate::{execution_unit::ExecutionUnit, tensor::Error, Instruction};

use super::{
    stream::{Stream, StreamState},
    transaction::{get_instruction_transactions, Transaction},
};

/// Simulate an execution of streams and emit operand transactions.
#[allow(unused)]
pub fn spawn_and_join_streams(
    streams: &[Stream],
    instructions: &[(Vec<usize>, Vec<usize>)],
    max_concurrent_streams: usize,
) -> Vec<Transaction> {
    let mut actual_transactions = vec![];
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
            let stream_instructions = &streams[stream_to_spawn].instructions;
            for instruction in stream_instructions.iter() {
                let instruction = *instruction;
                let (inputs, outputs) = &instructions[instruction];
                let mut instruction_transactions =
                    get_instruction_transactions(instruction, inputs, outputs);
                actual_transactions.extend_from_slice(&mut instruction_transactions);
            }
            // Immediately join the thread.
            joined_streams.insert(stream_to_spawn);
        }
    }
    actual_transactions
}

fn join_stream(
    stream: usize,
    streams: &mut Vec<Stream>,
    _threads: &mut Vec<Option<JoinHandle<Result<(), Error>>>>,
    active_streams: &mut BTreeSet<usize>,
) -> Result<(), Error> {
    debug_assert_eq!(StreamState::Spawned, streams[stream].state);
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
    let new_state = StreamState::Joined;
    #[cfg(feature = "verbose_streams")]
    println!(
        "Transition stream {}  {} -> {}",
        stream, streams[stream].state, new_state
    );
    streams[stream].state = new_state;
    active_streams.remove(&stream);
    #[cfg(feature = "verbose_streams")]
    println!("active_streams {}", active_streams.len());
    Ok(())
}

fn spawn_stream(
    stream: usize,
    streams: &mut Vec<Stream>,
    _threads: &mut Vec<Option<JoinHandle<Result<(), Error>>>>,
    instructions: &Arc<Vec<Instruction>>,
    active_streams: &mut BTreeSet<usize>,
) -> Result<(), Error> {
    debug_assert_eq!(StreamState::Unreached, streams[stream].state);
    let new_state = StreamState::Spawned;
    #[cfg(feature = "verbose_streams")]
    println!(
        "Transition stream {}  {} -> {}",
        stream, streams[stream].state, new_state
    );
    streams[stream].state = new_state;
    active_streams.insert(stream);
    #[cfg(feature = "verbose_streams")]
    println!("active_streams {}", active_streams.len());

    if instructions.len() == 0 {
        return Ok(());
    }

    let stream_instructions = streams[stream].instructions.clone();
    let instructions = instructions.clone();

    ExecutionUnit::execute(stream_instructions, instructions)?;
    /*
    let spawned_thread =
        thread::spawn(|| ExecutionUnit::execute(stream_instructions, instructions));
    threads[stream] = Some(spawned_thread);
    */
    Ok(())
}

pub fn execute_streams(
    streams: &mut Vec<Stream>,
    instructions: &Arc<Vec<Instruction>>,
    max_concurrent_streams: usize,
) -> Result<(), Error> {
    let mut threads: Vec<Option<JoinHandle<Result<(), Error>>>> = vec![];
    for _ in 0..streams.len() {
        threads.push(None);
    }
    let range = 0..streams.len();
    let mut active_streams = BTreeSet::new();

    let mut unreached_streams = (0..streams.len()).collect::<BTreeSet<_>>();
    while unreached_streams.len() != 0 {
        let mut spawned_streams = vec![];
        for i in unreached_streams.iter() {
            let is_unreached = streams[*i].state == StreamState::Unreached;
            if is_unreached {
                // Join each dependency
                let n = streams[*i].dependencies.len();
                let mut all_dependencies_are_joined = true;
                for j in 0..n {
                    let dependency = streams[*i].dependencies[j];
                    if streams[dependency].state == StreamState::Spawned {
                        join_stream(dependency, streams, &mut threads, &mut active_streams)?;
                    } else if streams[dependency].state == StreamState::Joined {
                        #[cfg(feature = "verbose_streams")]
                        println!(
                            "note stream {} is already {}",
                            dependency,
                            StreamState::Joined
                        );
                    } else {
                        all_dependencies_are_joined = false;
                    }
                }

                if !all_dependencies_are_joined {
                    continue;
                }
                if active_streams.len() == max_concurrent_streams {
                    // Join the oldest active stream before spawning this one.
                    let oldest = active_streams.iter().min().map(|x| *x);
                    if let Some(oldest) = oldest {
                        join_stream(oldest, streams, &mut threads, &mut active_streams)?;
                    }
                }
                spawn_stream(*i, streams, &mut threads, instructions, &mut active_streams)?;
                spawned_streams.push(*i);
            } else {
                panic!("Can not spawn stream {} because it is not unreached", i);
            }
        }
        for spawned in spawned_streams.iter() {
            unreached_streams.remove(spawned);
        }
    }
    for i in range {
        if streams[i].state == StreamState::Spawned {
            join_stream(i, streams, &mut threads, &mut active_streams)?;
        }
    }
    debug_assert_eq!(0, active_streams.len());

    Ok(())
}

pub fn reset_streams(streams: &mut Vec<Stream>) {
    for i in 0..streams.len() {
        streams[i].state = StreamState::Unreached;
    }
}
