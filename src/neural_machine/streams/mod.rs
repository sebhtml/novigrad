use std::{collections::BTreeSet, fmt::Display};

use crate::{execution_unit::ExecutionUnit, tensor::Error, Instruction};
#[cfg(test)]
mod tests;

pub struct Stream {
    pub id: usize,
    pub state: StreamState,
    pub dependencies: Vec<usize>,
    pub instructions: Vec<usize>,
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
pub fn make_streams(instruction_operands: &[(Vec<usize>, Vec<usize>)]) -> Vec<Stream> {
    // A list of dependencies for each instruction.
    let instruction_dependencies = get_instruction_dependencies(instruction_operands);

    #[cfg(feature = "verbose_streams")]
    for (i, i_dependencies) in instruction_dependencies.iter().enumerate() {
        println!(
            "[assign_streams] INSTRUCTION_DEPENDENCIES  instruction: {},  instructions: {}",
            i,
            i_dependencies
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let instruction_streams = assign_instructions_to_streams(&instruction_dependencies);

    #[cfg(feature = "verbose_streams")]
    {
        println!("Instruction streams");
        for (i, stream) in instruction_streams.iter().enumerate() {
            println!("Instruction {}  stream {}", i, stream);
        }
    }

    let max_stream = instruction_streams.iter().max();
    let stream_count = match max_stream {
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
            instructions,
        };
        streams.push(stream);
    }

    // Assign stream dependencies
    for i in 0..streams.len() {
        let stream_instructions = &streams[i].instructions;
        let first_instruction = stream_instructions[0];
        let dependency_instructions = &instruction_dependencies[first_instruction];
        let dependency_streams = dependency_instructions
            .iter()
            .map(|i| instruction_streams[*i])
            .collect::<Vec<_>>();
        streams[i].dependencies = dependency_streams;
    }

    #[cfg(feature = "verbose_streams")]
    for stream in streams.iter() {
        println!("STREAM {}", stream);
    }

    streams
}

fn get_instruction_dependencies(instructions: &[(Vec<usize>, Vec<usize>)]) -> Vec<Vec<usize>> {
    let mut dependencies = vec![vec![]; instructions.len()];
    for (i, (i_inputs, i_outputs)) in instructions.iter().enumerate() {
        //----------------------------
        // Write(operandX) and Read(operandX) must not be not re-ordered.
        //----------------------------

        // Read of input depends on previous write.
        for i_input in i_inputs.iter() {
            // find the closest prior instruction that writes to his operand.
            // Then add it to the dependencies.
            let j_range = 0..i;
            for j in j_range.rev() {
                let j_outputs = &instructions[j].1;

                if j_outputs.contains(&i_input) {
                    dependencies[i].push(j);
                    break;
                }
            }
        }

        //----------------------------
        // Write(operandX) and Write(operandX) must not be not re-ordered.
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
                    dependencies[i].push(j);
                    break;
                }
            }
        }

        //----------------------------
        // Read(operandX) and Write(operandX) must not be not re-ordered.
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
                    dependencies[i].push(j);
                    break;
                }
            }
        }

        dependencies[i].sort();
        dependencies[i].dedup();
    }
    dependencies
}

fn assign_instructions_to_streams(instruction_dependencies: &[Vec<usize>]) -> Vec<usize> {
    let no_stream = usize::MAX;
    let n = instruction_dependencies.len();
    let mut instruction_streams: Vec<usize> = vec![no_stream; n];
    let mut next_stream = 0;
    for (i_inst, i_deps) in instruction_dependencies.iter().enumerate() {
        if i_deps.len() == 1 {
            let dependency_instruction = i_deps[0];
            let stream = instruction_streams[dependency_instruction];
            if stream == no_stream {
                panic!("Prior instruction has no assigned stream");
            }
            instruction_streams[i_inst] = stream;
        } else {
            // Instructions with 2 or more dependencies can wait for their dependencies,
            // and then run.
            instruction_streams[i_inst] = next_stream;
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

fn join_stream(stream: usize, streams: &mut Vec<Stream>, active_streams: &mut BTreeSet<usize>) {
    debug_assert_eq!(StreamState::Spawned, streams[stream].state);
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
}

fn spawn_stream(
    stream: usize,
    streams: &mut Vec<Stream>,
    instructions: &[Instruction],
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

    let stream_instructions = &streams[stream].instructions;
    ExecutionUnit::execute(stream_instructions, instructions)
}

pub fn execute_streams(
    streams: &mut Vec<Stream>,
    instructions: &[Instruction],
    max_concurrent_streams: usize,
) -> Result<(), Error> {
    let range = 0..streams.len();
    let mut active_streams = BTreeSet::new();
    for i in range.clone().into_iter() {
        if streams[i].state == StreamState::Unreached {
            // Join each dependency
            let n = streams[i].dependencies.len();
            for j in 0..n {
                let dependency = streams[i].dependencies[j];
                if streams[dependency].state == StreamState::Spawned {
                    join_stream(dependency, streams, &mut active_streams);
                } else if streams[dependency].state == StreamState::Joined {
                    #[cfg(feature = "verbose_streams")]
                    println!(
                        "note stream {} is already {}",
                        dependency,
                        StreamState::Joined
                    );
                } else {
                    panic!("Can not join unspawned stream {}", dependency);
                }
            }

            if active_streams.len() == max_concurrent_streams {
                // Join the oldest active stream before spawning this one.
                let oldest = active_streams.iter().min().map(|x| *x);
                if let Some(oldest) = oldest {
                    join_stream(oldest, streams, &mut active_streams);
                }
            }
            spawn_stream(i, streams, instructions, &mut active_streams)?;
        } else {
            panic!("Can not spawn stream {} because it is not unreached", i);
        }
    }
    for i in range {
        if streams[i].state == StreamState::Spawned {
            join_stream(i, streams, &mut active_streams);
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
