#[cfg(test)]
mod tests;

pub mod instruction;
pub mod queue;
pub mod scheduler;
pub mod stream;
pub mod transaction;

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
