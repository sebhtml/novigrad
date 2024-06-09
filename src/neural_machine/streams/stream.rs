use std::{collections::BTreeSet, fmt::Display, sync::Arc};

use super::instruction::{get_instruction_dependencies, Dependencies};

pub const STREAM_NONE: usize = usize::MAX;

pub struct Stream {
    pub id: usize,
    pub dependencies: Vec<usize>,
    pub instructions: Arc<Vec<usize>>,
}

impl Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = write!(f, "Stream id: {}", self.id,);
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

pub fn print_streams(name: &str, streams: &[Stream]) {
    println!("Streams  Description: {}  Count: {}", name, streams.len());
    for (i, stream) in streams.iter().enumerate() {
        println!(
            "stream: {}  dependencies_len: {}  instructions_len: {}  dependencies: {:?}   instructions: {:?}",
            i, stream.dependencies.len(),stream.instructions.len(), stream.dependencies,  stream.instructions
        )
    }
}

/// Group <N> instructions in <M> streams using a dependency analysis.
pub fn make_streams(
    instructions: &[(Vec<usize>, Vec<usize>)],
    minimum_write_before_read_for_new_stream: usize,
    minimum_dependents_for_stream: usize,
    minimum_stream_instructions: usize,
) -> Vec<Stream> {
    // A list of dependencies for each instruction.
    let instruction_dependencies = get_instruction_dependencies(instructions);

    //#[cfg(feature = "verbose_streams")]
    for (i, i_dependencies) in instruction_dependencies.iter().enumerate() {
        println!(
            "[assign_streams] INSTRUCTION_DEPENDENCIES  instruction: {},  write_before_read: {:?},  read_before_write: {:?},  write_before_write: {:?}",
            i,
            i_dependencies.write_before_read_dependencies,
            i_dependencies.read_before_write_dependencies,
            i_dependencies.write_before_write_dependencies,
        );
    }

    let instruction_streams = assign_instructions_to_streams(
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
        &instruction_dependencies,
    );

    #[cfg(feature = "verbose_streams")]
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
            dependencies: Default::default(),
            instructions: instructions.into(),
        };
        streams.push(stream);
    }

    // Assign stream dependencies
    for i in 0..streams.len() {
        // The dependencies of a stream are the assigned streams of all the instructions it contains.
        let stream_instructions = &streams[i].instructions;
        let mut dependency_streams = stream_instructions
            .iter()
            .map(|instruction| {
                let dependency_instructions =
                    &instruction_dependencies[*instruction].all_dependencies;
                dependency_instructions
                    .iter()
                    .map(|instruction| instruction_streams[*instruction])
                    .filter(|x| *x != i)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .concat();
        dependency_streams.sort();
        dependency_streams.dedup();
        streams[i].dependencies = dependency_streams;
    }

    streams
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
    minimum_dependents_for_stream: usize,
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
    // So we can compute the inputs of Concat in parallel basically.
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

    // Assign a stream to any instruction that have more than N dependents
    for (i, deps) in instruction_dependencies.iter().enumerate() {
        if !instructions_with_no_stream.contains(&i) {
            continue;
        }
        let dependents = &deps.all_dependents;
        //println!("DEBUG instruction: {}  dependents: {:?}", i, dependents);
        if dependents.len() >= minimum_dependents_for_stream {
            instruction_streams[i] = next_stream;
            next_stream += 1;
            instructions_with_no_stream.remove(&i);
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

    //#[cfg(feature = "verbose_streams")]
    for (i_inst, i_stream) in instruction_streams.iter().enumerate() {
        println!(
            "[assign_streams] INSTRUCTION_STREAM  instruction: {},  stream: {}",
            i_inst, i_stream,
        );
    }

    instruction_streams
}
