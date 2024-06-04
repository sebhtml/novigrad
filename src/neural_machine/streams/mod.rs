use std::{
    collections::{BTreeSet, HashSet},
    fmt::Display,
    sync::Arc,
    thread::JoinHandle,
};

use crate::{execution_unit::ExecutionUnit, tensor::Error, Instruction};

#[cfg(test)]
mod tests;

const STREAM_NONE: usize = usize::MAX;

pub struct Stream {
    pub id: usize,
    pub state: StreamState,
    pub dependencies: Vec<usize>,
    pub instructions: Arc<Vec<usize>>,
}

/// Maker sure that no instruction writes to machine inputs.
pub fn verify_machine_inputs(machine_inputs: &[usize], instructions: &[(Vec<usize>, Vec<usize>)]) {
    for machine_input in machine_inputs {
        for (i, (_, outputs)) in instructions.iter().enumerate() {
            if outputs.contains(machine_input) {
                panic!(
                    "[assign_streams] PROBLEM-0001 instruction {} writes ot machine input {} !",
                    i, machine_input
                );
            }
        }
    }
}

fn get_constants(instructions: &[(Vec<usize>, Vec<usize>)]) -> HashSet<usize> {
    let outputs = instructions
        .iter()
        .map(|(_, outputs)| outputs.clone())
        .collect::<Vec<Vec<usize>>>();
    let outputs = outputs.concat().into_iter().collect::<HashSet<_>>();
    let mut inputs = instructions
        .iter()
        .map(|(inputs, _)| inputs.clone())
        .collect::<Vec<_>>()
        .concat();
    inputs.sort();
    inputs.dedup();
    let constants = inputs
        .into_iter()
        .filter(|x| !outputs.contains(x))
        .collect::<HashSet<_>>();
    constants
}

/// Group <N> instructions in <M> streams using a dependency analysis.
pub fn make_streams(
    instructions: &[(Vec<usize>, Vec<usize>)],
    minimum_write_before_read_for_new_stream: usize,
) -> Vec<Stream> {
    let constants = get_constants(instructions);
    for constant in constants.iter() {
        println!("CONSTANT: {}", constant);
    }
    for (i, (inputs, outputs)) in instructions.iter().enumerate() {
        println!(
            "INSTRUCTION {}  inputs {:?}  outputs {:?}",
            i, inputs, outputs
        );
    }
    // A list of dependencies for each instruction.
    let instruction_dependencies = get_instruction_dependencies(instructions);

    #[cfg(feature = "verbose_streams")]
    for (i, i_dependencies) in instruction_dependencies.iter().enumerate() {
        println!(
            "[assign_streams] INSTRUCTION_DEPENDENCIES  instruction: {},  write_before_read: {:?},  read_before_write: {:?},  write_before_write: {:?}",
            i,
            i_dependencies.write_before_read,
            i_dependencies.read_before_write,
            i_dependencies.write_before_write,
        );
    }

    let instruction_streams = assign_instructions_to_streams(
        minimum_write_before_read_for_new_stream,
        &instruction_dependencies,
    );

    #[cfg(feature = "verbose_streams")]
    {
        for (i, stream) in instruction_streams.iter().enumerate() {
            println!("Instruction {}  stream {}", i, stream);
        }
    }

    let max_stream = instruction_streams.iter().max();
    let stream_count = match max_stream {
        Some(&usize::MAX) => panic!(),
        Some(max_stream) => max_stream + 1,
        None => 0,
    };
    let mut stream_instructions = vec![vec![]; stream_count];
    for (i_inst, i_stream) in instruction_streams.iter().enumerate() {
        stream_instructions[*i_stream].push(i_inst);
    }

    let mut streams: Vec<Stream> = vec![];
    for i in 0..stream_instructions.len() {
        let instructions = stream_instructions[i].clone();
        let stream = Stream {
            id: i,
            state: Default::default(),
            dependencies: Default::default(),
            instructions: instructions.into(),
        };
        streams.push(stream);
    }

    // Assign stream dependencies
    for i in 0..streams.len() {
        let stream_instructions = &streams[i].instructions;
        let first_instruction = stream_instructions[0];
        let dependency_instructions = &instruction_dependencies[first_instruction].dependencies;
        let mut dependency_streams = dependency_instructions
            .iter()
            .map(|i| instruction_streams[*i])
            .collect::<Vec<_>>();
        dependency_streams.sort();
        dependency_streams.dedup();
        streams[i].dependencies = dependency_streams;
    }

    #[cfg(feature = "verbose_streams")]
    for stream in streams.iter() {
        println!("STREAM {}", stream);
    }

    streams
}

fn get_instruction_dependencies(instructions: &[(Vec<usize>, Vec<usize>)]) -> Vec<Dependencies> {
    let mut dependencies = vec![Dependencies::default(); instructions.len()];
    for (i, (i_inputs, i_outputs)) in instructions.iter().enumerate() {
        //----------------------------
        // Write(operandX) and Read(operandX) must not be re-ordered.
        //----------------------------

        // Read of input depends on previous write.
        for i_input in i_inputs.iter() {
            // find the closest prior instruction that writes to his operand.
            // Then add it to the dependencies.
            let j_range = 0..i;
            for j in j_range.rev() {
                let j_outputs = &instructions[j].1;

                if j_outputs.contains(&i_input) {
                    dependencies[i].write_before_read.push(j);
                    break;
                }
            }
        }

        //----------------------------
        // Write(operandX) and Write(operandX) must not be re-ordered.
        //----------------------------

        // Write of output depends on previous write.
        // If previous instruction j writes to the same output as instruction i,
        // the order of the writes must be preserved.
        for i_output in i_outputs.iter() {
            // find the closest prior instruction that writes to his operand.
            // Then add it to the dependencies.
            let j_range = 0..i;
            for j in j_range.rev() {
                let j_outputs = &instructions[j].1;

                if j_outputs.contains(&i_output) {
                    dependencies[i].write_before_write.push(j);
                    break;
                }
            }
        }

        //----------------------------
        // Read(operandX) and Write(operandX) must not be re-ordered.
        //----------------------------
        // Write of output depends on previous read.
        // If we write to the operand before the previous read,
        // then the previous read will read a bad value from the future.
        // If previous instruction j reads to the same output as instruction i,
        // the order of the read-then-write must be preserved.
        for i_output in i_outputs.iter() {
            // find the closest prior instruction that writes to his operand.
            // Then add it to the dependencies.
            let j_range = 0..i;
            for j in j_range.rev() {
                let j_inputs = &instructions[j].0;

                if j_inputs.contains(&i_output) {
                    dependencies[i].read_before_write.push(j);
                    break;
                }
            }
        }
    }

    for entry in dependencies.iter_mut() {
        let mut deps = vec![
            entry.write_before_read.clone(),
            entry.write_before_write.clone(),
            entry.read_before_write.clone(),
        ]
        .concat();
        deps.sort();
        deps.dedup();
        entry.dependencies = deps;
    }

    for dependent in 0..dependencies.len() {
        for dependency in dependencies[dependent].dependencies.clone().iter() {
            dependencies[*dependency].dependents.push(dependent);
        }
    }

    dependencies
}

fn assign_instructions_to_streams(
    minimum_write_before_read_for_new_stream: usize,
    instruction_dependencies: &[Dependencies],
) -> Vec<usize> {
    let n = instruction_dependencies.len();
    let mut instructions_with_no_stream = (0..n).collect::<BTreeSet<_>>();
    let mut instruction_streams: Vec<usize> = vec![STREAM_NONE; n];
    let mut next_stream = 0;

    // Assign streams when an instruction has more than N inputs.
    // Gemm has 3 inputs.
    // Concat has N inputs.
    // So we compute the inputs of Concat in parallel basically.
    for (i, deps) in instruction_dependencies.iter().enumerate() {
        if !instructions_with_no_stream.contains(&i) {
            continue;
        }
        let write_before_read = deps.write_before_read.len();
        if write_before_read >= minimum_write_before_read_for_new_stream {
            let mut all_dependencies_have_no_stream = true;
            for j in deps.write_before_read.iter() {
                if !instructions_with_no_stream.contains(j) {
                    all_dependencies_have_no_stream = false;
                    break;
                }
            }

            if !all_dependencies_have_no_stream {
                continue;
            }
            for j in deps.write_before_read.iter() {
                instruction_streams[*j] = next_stream;
                next_stream += 1;
                instructions_with_no_stream.remove(j);
            }
        }
    }

    while let Some(instruction) = instructions_with_no_stream.pop_first() {
        let mut streams = instruction_dependencies[instruction]
            .dependencies
            .iter()
            .map(|i| instruction_streams[*i].clone())
            .collect::<Vec<_>>();
        streams.sort();
        streams.dedup();

        if streams.len() == 1 && streams[0] != STREAM_NONE {
            instruction_streams[instruction] = streams[0];
        }

        if instruction_streams[instruction] == STREAM_NONE {
            instruction_streams[instruction] = next_stream;
            next_stream += 1;
        }
    }

    #[cfg(feature = "verbose_streams")]
    for (i_inst, i_stream) in instruction_streams.iter().enumerate() {
        println!(
            "[assign_streams] INSTRUCTION_STREAM  instruction: {},  stream: {}",
            i_inst, i_stream,
        );
    }

    instruction_streams
}

impl Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = write!(f, "Stream id: {}", self.id,);
        let _ = write!(f, "\n");
        let _ = write!(f, "state: {}", self.state,);
        let _ = write!(f, "\n");
        let _ = write!(f, "dependencies_len: {}", self.dependencies.len(),);
        let _ = write!(f, "\n");
        let _ = write!(f, "dependencies: {:?}", self.dependencies,);
        let _ = write!(f, "\n");
        let _ = write!(f, "instructions_len: {:?}", self.instructions.len(),);
        let _ = write!(f, "\n");
        let _ = write!(f, "instructions: {:?}", self.instructions,);
        let result = write!(f, "\n");
        result
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum StreamState {
    Unreached,
    Spawned,
    Joined,
}

impl Into<String> for &StreamState {
    fn into(self) -> String {
        match self {
            StreamState::Unreached => "Unreached",
            StreamState::Spawned => "Spawned",
            StreamState::Joined => "Joined",
        }
        .into()
    }
}

impl Display for StreamState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let as_string: String = self.into();
        let result = write!(f, "{}", as_string);
        result
    }
}

impl Default for StreamState {
    fn default() -> Self {
        StreamState::Unreached
    }
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

pub fn make_simple_instructions(instructions: &Vec<Instruction>) -> Vec<(Vec<usize>, Vec<usize>)> {
    let instructions = instructions
        .iter()
        .map(|instruction| {
            let inputs = instruction
                .inputs()
                .iter()
                .map(|x| x.name())
                .collect::<Vec<_>>();
            let outputs = instruction
                .outputs()
                .iter()
                .map(|x| x.name())
                .collect::<Vec<_>>();
            (inputs, outputs)
        })
        .collect::<Vec<_>>();
    instructions
}

#[derive(Clone, Debug)]
pub struct Dependencies {
    pub write_before_read: Vec<usize>,
    pub write_before_write: Vec<usize>,
    pub read_before_write: Vec<usize>,
    pub dependencies: Vec<usize>,
    pub dependents: Vec<usize>,
}

impl Default for Dependencies {
    fn default() -> Self {
        Self {
            write_before_read: Default::default(),
            write_before_write: Default::default(),
            read_before_write: Default::default(),
            dependencies: Default::default(),
            dependents: Default::default(),
        }
    }
}

pub fn print_streams(name: &str, streams: &[Stream]) {
    println!("Streams  Description: {}  Count: {}", name, streams.len());
    for (i, stream) in streams.iter().enumerate() {
        println!(
            "stream {},  dependencies {:?}  instructions {:?}",
            i, stream.dependencies, stream.instructions
        )
    }
}
