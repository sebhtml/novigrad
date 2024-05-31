use std::{collections::HashMap, fmt::Display};

pub struct Stream {
    pub id: usize,
    pub state: StreamState,
    pub dependencies: Vec<usize>,
    pub instructions: Vec<usize>,
}

/// Maker sure that no instruction writes to machine inputs.
fn verify_machine_inputs(machine_inputs: &[usize], instructions: &[(Vec<usize>, Vec<usize>)]) {
    for machine_input in machine_inputs {
        for (i, (_, outputs)) in instructions.iter().enumerate() {
            if outputs.contains(machine_input) {
                println!(
                    "[assign_streams] PROBLEM-0001 instruction {} writes ot machine input {} !",
                    i, machine_input
                );
            }
        }
    }
}

/// Group <N> instructions in <M> streams using a dependency analysis.
pub fn make_streams(
    machine_inputs: &[usize],
    instruction_operands: &[(Vec<usize>, Vec<usize>)],
) -> Vec<Stream> {
    verify_machine_inputs(machine_inputs, instruction_operands);
    // A list of dependencies (instructions) for each instruction.
    let dependencies = get_instruction_instruction_dependencies(instruction_operands);

    for (_i, (i_inputs, _i_outputs)) in instruction_operands.iter().enumerate() {
        for i_input in i_inputs.iter() {
            if machine_inputs.contains(i_input) {
                // Nothing to do since the input is ready.
                continue;
            }
        }
    }

    for (i, i_dependencies) in dependencies.iter().enumerate() {
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

    let instruction_streams = make_instruction_streams(&dependencies);

    for (stream, instructions) in instruction_streams.iter().enumerate() {
        println!(
            "[assign_streams] STREAM  stream: {},  instructions: {}",
            stream,
            instructions
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let mut len_distribution = HashMap::new();
    for instructions in instruction_streams.iter() {
        let len = instructions.len();
        let value = len_distribution.entry(len).or_insert(0);
        *value += 1;
    }
    for (len, streams) in len_distribution {
        println!(
            "[assign_streams] DISTRIBUTION  length: {}  streams: {}",
            len, streams
        );
    }

    let stream_dependency_streams = vec![vec![]; instruction_streams.len()];

    let mut streams: Vec<Stream> = vec![];
    for i in 0..instruction_streams.len() {
        let instructions = instruction_streams[i].clone();
        let stream = Stream {
            id: i,
            state: Default::default(),
            dependencies: stream_dependency_streams[i].clone(),
            instructions,
        };
        streams.push(stream);
    }

    for i in 0..streams.len() {
        let i_instructions = &streams[i].instructions;
        let i_first_instruction = i_instructions[0];
        let i_inputs = &instruction_operands[i_first_instruction].0;
        for i_input in i_inputs {
            if i > 0 {
                // find the closest prior instruction that writes to his operand.
                // Then add it to the dependencies.
                let j_range = 0..(i - 1);
                for j in j_range.rev() {
                    let j_instructions = &streams[j].instructions;
                    let j_last_instruction = j_instructions[j_instructions.len() - 1];
                    let j_outputs = &instruction_operands[j_last_instruction].1;

                    if j_outputs.contains(&i_input) {
                        streams[i].dependencies.push(j);
                        break;
                    }
                }
            }
        }
    }
    streams
}

fn get_instruction_instruction_dependencies(
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<Vec<usize>> {
    let mut dependencies = vec![];
    for (i, (i_inputs, _i_outputs)) in instructions.iter().enumerate() {
        let mut i_dependencies = vec![];
        for i_input in i_inputs.iter() {
            if i > 0 {
                // find the closest prior instruction that writes to his operand.
                // Then add it to the dependencies.
                let j_range = 0..(i - 1);
                for j in j_range.rev() {
                    let j_outputs = &instructions[j].1;

                    if j_outputs.contains(&i_input) {
                        i_dependencies.push(j);
                        break;
                    }
                }
            }
        }
        dependencies.push(i_dependencies);
    }
    dependencies
}

fn make_instruction_streams(instruction_dependencies: &[Vec<usize>]) -> Vec<Vec<usize>> {
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

    println!(
        "[assign_streams] REDUCTION  instructions: {},  streams: {}",
        instruction_dependencies.len(),
        next_stream
    );
    for (i_inst, i_stream) in instruction_streams.iter().enumerate() {
        println!(
            "[assign_streams] INSTRUCTION_STREAM  instruction: {},  stream: {}",
            i_inst, i_stream,
        );
    }
    let mut streams = vec![vec![]; next_stream];
    for (i_inst, i_stream) in instruction_streams.iter().enumerate() {
        streams[*i_stream].push(i_inst);
    }

    streams
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

#[derive(Clone, PartialEq)]
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

pub fn simulate_execution(streams: &mut Vec<Stream>) {
    let range = 0..streams.len();
    for i in range {
        if streams[i].state == StreamState::Unreached {
            // Join each dependency
            let n = streams[i].dependencies.len();
            for j in 0..n {
                let dependency = streams[i].dependencies[j];
                if streams[dependency].state == StreamState::Spawned {
                    println!("Transition stream {} {}", dependency, StreamState::Joined);
                    streams[dependency].state = StreamState::Joined;
                } else {
                    panic!("Can not join unspawned stream {}", dependency);
                }
            }
            println!("Transition stream {} {}", i, StreamState::Spawned);
            streams[i].state = StreamState::Spawned;
        } else {
            panic!("Can not spawn stream {} because it is not unreached", i);
        }
    }
}
