pub fn assign_streams(
    machine_inputs: &[usize],
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<usize> {
    // A list of dependencies (instructions) for each instruction.
    let mut dependencies: Vec<Vec<usize>> = vec![vec![]; instructions.len()];
    for (i, (i_inputs, _i_outputs)) in instructions.iter().enumerate() {
        for i_input in i_inputs.iter() {
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
                println!("[assign_streams] Could not find the closest prior instruction that writes to operand {}, which is an input of instruction {}",
                        i_input, i);
            }
        }
    }
    vec![]
}
