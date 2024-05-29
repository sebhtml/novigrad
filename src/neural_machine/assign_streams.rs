fn discover_constants(instructions: &[(Vec<usize>, Vec<usize>)]) -> Vec<usize> {
    let mut inputs = instructions
        .iter()
        .map(|(inputs, _)| inputs)
        .flatten()
        .collect::<Vec<_>>();
    inputs.sort();
    inputs.dedup();

    let mut outputs = instructions
        .iter()
        .map(|(_, outputs)| outputs)
        .flatten()
        .collect::<Vec<_>>();
    outputs.sort();
    outputs.dedup();

    let constants = inputs
        .iter()
        .filter(|input| match outputs.binary_search(input) {
            Ok(_) => false,
            Err(_) => true,
        })
        .map(|x| **x)
        .collect::<Vec<_>>();
    constants
}

fn verify_machine_inputs(constants: &[usize], machine_inputs: &[usize]) {
    for machine_input in machine_inputs {
        if !constants.contains(machine_input) {
            println!(
                "[assign_streams] PROBLEM-0001 machine input {} should be a constant but it's not !",
                machine_input
            );
        }
    }
}

pub fn assign_streams(
    machine_inputs: &[usize],
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<usize> {
    let constants = discover_constants(instructions);
    println!("[assign_streams] there are {} constants", constants.len());
    verify_machine_inputs(&constants, machine_inputs);
    // A list of dependencies (instructions) for each instruction.
    let mut dependencies: Vec<Vec<usize>> = vec![vec![]; instructions.len()];
    for (i, (i_inputs, _i_outputs)) in instructions.iter().enumerate() {
        for i_input in i_inputs.iter() {
            if constants.contains(i_input) {
                continue;
            }
            let mut i_input_is_satisfied = false;
            if machine_inputs.contains(i_input) {
                // Nothing to do since the input is ready.
                i_input_is_satisfied = true;
            } else {
                // find the closest prior instruction that writes to his operand.
                // Then add it to the dependencies.
                if i > 0 {
                    let j_range = 0..(i - 1);
                    for j in j_range.rev() {
                        let j_outputs = &instructions[j].1;

                        if j_outputs.contains(&i_input) {
                            dependencies[i].push(j);
                            i_input_is_satisfied = true;
                            break;
                        }
                    }
                }
            }
            if !i_input_is_satisfied {
                println!("[assign_streams] PROBLEM-0002 Could not find the closest prior instruction that writes to operand {}, which is an input of instruction {}",
                        i_input, i);
            }
        }
    }
    vec![]
}
