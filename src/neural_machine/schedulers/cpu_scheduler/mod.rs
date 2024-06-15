#[cfg(test)]
mod tests;

mod controller;
mod execution_unit;
pub mod queue;
pub mod scheduler;

pub enum Command {
    Execute,
    Stop,
    WorkUnitDispatch(usize),
    WorkUnitCompletion(usize),
    ExecutionCompletion,
}
