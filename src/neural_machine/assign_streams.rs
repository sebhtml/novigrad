use std::collections::HashMap;

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
pub fn assign_streams(
    machine_inputs: &[usize],
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<Vec<usize>> {
    verify_machine_inputs(machine_inputs, instructions);
    // A list of dependencies (instructions) for each instruction.
    let dependencies = get_instruction_instruction_dependencies(instructions);

    for (_i, (i_inputs, _i_outputs)) in instructions.iter().enumerate() {
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

    let streams = make_instruction_streams(&dependencies);

    for (stream, instructions) in streams.iter().enumerate() {
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
    for instructions in streams.iter() {
        let len = instructions.len();
        let value = len_distribution.entry(len).or_insert(0);
        *value  += 1;
    }
    for (len, streams) in len_distribution {
        println!("[assign_streams] DISTRIBUTION  length: {}  streams: {}", len, streams);
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

    println!("[assign_streams] REDUCTION  instructions: {},  streams: {}", instruction_dependencies.len(), next_stream);
    for (i_inst, i_stream) in instruction_streams.iter().enumerate() {
        println!(
            "[assign_streams] INSTRUCTION_STREAM  instruction: {},  stream: {}",
    i_inst,
    i_stream,
        );
    }
    let mut streams = vec![vec![]; next_stream];
    for (i_inst, i_stream) in instruction_streams.iter().enumerate() {
        streams[*i_stream].push(i_inst);
    }
    streams
}