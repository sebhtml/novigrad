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

pub fn assign_streams(
    machine_inputs: &[usize],
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<usize> {
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
            "[assign_streams] DEPENDENCIES  instruction: {},  dependencies: {}",
            i,
            i_dependencies
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    vec![]
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
