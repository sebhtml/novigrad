use crate::Instruction;

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

pub fn get_instruction_dependencies(
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<Dependencies> {
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

#[allow(unused)]
pub fn print_instructions(instructions: &[(Vec<usize>, Vec<usize>)]) {
    for (i, (inputs, outputs)) in instructions.iter().enumerate() {
        println!(
            "INSTRUCTION {}  inputs {:?}  outputs {:?}",
            i, inputs, outputs
        );
    }
}
