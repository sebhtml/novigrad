use std::{collections::BTreeSet, fmt::Display, sync::Arc, thread::JoinHandle};

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

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum Access {
    Read,
    Write,
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct Transaction {
    pub instruction: usize,
    pub operand: usize,
    pub access: Access,
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

/// Group <N> instructions in <M> streams using a dependency analysis.
pub fn make_streams(
    instructions: &[(Vec<usize>, Vec<usize>)],
    minimum_write_before_read_for_new_stream: usize,
    minimum_stream_instructions: usize,
) -> Vec<Stream> {
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
        minimum_stream_instructions,
        &instruction_dependencies,
    );

    //#[cfg(feature = "verbose_streams")]
    {
        for (i, stream) in instruction_streams.iter().enumerate() {
            println!("STREAM-ASSIGNMENT Instruction {}  stream {}", i, stream);
        }
    }

    let max_stream = instruction_streams.iter().max();
    let stream_count = match max_stream {
        Some(&usize::MAX) => {
            println!("Instruction streams:");
            for (i, stream) in instruction_streams.iter().enumerate() {
                println!("Instruction {}  stream {}", i, stream);
            }
            panic!();
        }
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
        let dependency_instructions = &instruction_dependencies[first_instruction].all_dependencies;
        let mut dependency_streams = dependency_instructions
            .iter()
            .map(|i| instruction_streams[*i])
            .collect::<Vec<_>>();
        dependency_streams.sort();
        dependency_streams.dedup();
        streams[i].dependencies = dependency_streams;
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
                    dependencies[i].write_before_read_dependencies.push(j);
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
                    dependencies[i].write_before_write_dependencies.push(j);
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
                    dependencies[i].read_before_write_dependencies.push(j);
                    break;
                }
            }
        }
    }

    for entry in dependencies.iter_mut() {
        let mut deps = vec![
            entry.write_before_read_dependencies.clone(),
            entry.write_before_write_dependencies.clone(),
            entry.read_before_write_dependencies.clone(),
        ]
        .concat();
        deps.sort();
        deps.dedup();
        entry.all_dependencies = deps;
    }

    for dependent in 0..dependencies.len() {
        for dependency in dependencies[dependent]
            .write_before_read_dependencies
            .clone()
            .iter()
        {
            dependencies[*dependency]
                .write_before_read_dependents
                .push(dependent);
        }
    }
    for dependent in 0..dependencies.len() {
        for dependency in dependencies[dependent]
            .write_before_write_dependencies
            .clone()
            .iter()
        {
            dependencies[*dependency]
                .write_before_write_dependents
                .push(dependent);
        }
    }
    for dependent in 0..dependencies.len() {
        for dependency in dependencies[dependent]
            .read_before_write_dependencies
            .clone()
            .iter()
        {
            dependencies[*dependency]
                .read_before_write_dependents
                .push(dependent);
        }
    }
    for dependent in 0..dependencies.len() {
        for dependency in dependencies[dependent].all_dependencies.clone().iter() {
            dependencies[*dependency].all_dependents.push(dependent);
        }
    }

    dependencies
}

/// Assign a stream to each instruction.
/// - `minimum_write_before_read_for_new_stream` dictates how instructions with many inputs are dealt with.
/// - `minimum_stream_instructions` dictates how many instructions should streams have at the end.
/// - `instruction_dependencies` dependencies between streams based on:
///   - write-before-read strict ordering
///   - read-before-write strict ordering
///   - write-before-write strict ordering
///
/// Returns assigned streams, in the same order as the input instructions.
fn assign_instructions_to_streams(
    minimum_write_before_read_for_new_stream: usize,
    minimum_stream_instructions: usize,
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
        let write_before_read = deps.write_before_read_dependencies.len();
        if write_before_read >= minimum_write_before_read_for_new_stream {
            let mut all_dependencies_have_no_stream = true;
            for j in deps.write_before_read_dependencies.iter() {
                if !instructions_with_no_stream.contains(j) {
                    all_dependencies_have_no_stream = false;
                    break;
                }
            }

            if !all_dependencies_have_no_stream {
                continue;
            }
            for j in deps.write_before_read_dependencies.iter() {
                instruction_streams[*j] = next_stream;
                next_stream += 1;
                instructions_with_no_stream.remove(j);
            }
        }
    }

    while let Some(instruction) = instructions_with_no_stream.pop_first() {
        // Take the instruction with no assigned stream and assign a stream based on the streams
        // of its dependency instructions
        let mut streams = instruction_dependencies[instruction]
            .all_dependencies
            .iter()
            .map(|i| instruction_streams[*i].clone())
            .collect::<Vec<_>>();
        streams.sort();
        streams.dedup();

        if streams.len() == 1 && streams[0] != STREAM_NONE {
            instruction_streams[instruction] = streams[0];
        }
    }

    for (instruction, _) in instruction_streams
        .iter()
        .enumerate()
        .filter(|(_, stream)| **stream == STREAM_NONE)
    {
        instructions_with_no_stream.insert(instruction);
    }

    let mut next_stream_instruction_count = 0;
    let mut last_added_instruction: Option<usize> = None;
    while let Some(instruction) = instructions_with_no_stream.pop_first() {
        // Reset next_stream if the last added instruction is not the previous one.
        if last_added_instruction != None && Some(instruction - 1) != last_added_instruction {
            next_stream += 1;
            next_stream_instruction_count = 0;
            last_added_instruction = None;
        }
        if instruction_streams[instruction] == STREAM_NONE
            && (last_added_instruction == None || Some(instruction - 1) == last_added_instruction)
        {
            instruction_streams[instruction] = next_stream;
            last_added_instruction = Some(instruction);
            next_stream_instruction_count += 1;
            if next_stream_instruction_count >= minimum_stream_instructions {
                next_stream += 1;
                next_stream_instruction_count = 0;
                last_added_instruction = None;
            }
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

/// Simulate an execution of streams and emit operand transactions.
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

fn get_instruction_transactions(
    instruction: usize,
    inputs: &[usize],
    outputs: &[usize],
) -> Vec<Transaction> {
    let mut transactions = vec![];
    for operand in inputs {
        let transaction = Transaction {
            instruction,
            operand: *operand,
            access: Access::Read,
        };
        transactions.push(transaction);
    }
    for operand in outputs {
        let transaction = Transaction {
            instruction,
            operand: *operand,
            access: Access::Write,
        };
        transactions.push(transaction);
    }
    transactions
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
    pub write_before_read_dependencies: Vec<usize>,
    pub write_before_write_dependencies: Vec<usize>,
    pub read_before_write_dependencies: Vec<usize>,
    pub all_dependencies: Vec<usize>,
    pub write_before_read_dependents: Vec<usize>,
    pub write_before_write_dependents: Vec<usize>,
    pub read_before_write_dependents: Vec<usize>,
    pub all_dependents: Vec<usize>,
}

impl Default for Dependencies {
    fn default() -> Self {
        Self {
            write_before_read_dependencies: Default::default(),
            write_before_write_dependencies: Default::default(),
            read_before_write_dependencies: Default::default(),
            all_dependencies: Default::default(),
            write_before_read_dependents: Default::default(),
            write_before_write_dependents: Default::default(),
            read_before_write_dependents: Default::default(),
            all_dependents: Default::default(),
        }
    }
}

pub fn print_streams(name: &str, streams: &[Stream]) {
    println!("Streams  Description: {}  Count: {}", name, streams.len());
    for (i, stream) in streams.iter().enumerate() {
        println!(
            "stream: {}  dependencies_len: {}  instructions_len: {}  dependencies: {:?}   instructions: {:?}",
            i, stream.dependencies.len(),stream.instructions.len(), stream.dependencies,  stream.instructions
        )
    }
}

#[allow(unused)]
pub fn print_instructions(instructions: &[(Vec<usize>, Vec<usize>)]) {
    for (i, (inputs, outputs)) in instructions.iter().enumerate() {
        println!(
            "INSTRUCTION {}  inputs {:?}  outputs {:?}",
            i, inputs, outputs
        );
    }
}
