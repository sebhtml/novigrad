#[cfg(test)]
pub mod tests;
mod train;
use std::{ops::Deref, time::SystemTime};

pub use train::*;
mod learning_tensor;
use crate::{devices::Device, Error, OptimizerTrait, Program, TensorF32};
pub use learning_tensor::*;

pub fn train(
    program: &Program,
    device: &Device,
    optimizer: &Box<dyn OptimizerTrait>,
    learning_rate: f32,
    epoch: usize,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
) -> Result<(), Error> {
    for i in 0..inputs.len() {
        train_back_propagation(
            program,
            device,
            optimizer,
            learning_rate,
            epoch,
            i,
            &inputs[i],
            &outputs[i],
        )?;
    }
    Ok(())
}

pub fn total_loss(program: &Program, inputs: &[Tensor], outputs: &[Tensor]) -> Result<f32, Error> {
    let mut total_error = 0.0;
    for i in 0..inputs.len() {
        let _ = program.forward(&[&inputs[i]])?;
        let expected_output = &outputs[i];
        let example_loss = program.loss(expected_output)?;
        let example_loss: &TensorF32 = &example_loss.tensor().deref().borrow();
        let example_loss: f32 = example_loss.try_into()?;
        total_error += example_loss;
    }

    Ok(total_error)
}

fn train_back_propagation(
    program: &Program,
    device: &Device,
    optimizer: &Box<dyn OptimizerTrait>,
    learning_rate: f32,
    _epoch: usize,
    _example_index: usize,
    x: &Tensor,
    y: &Tensor,
) -> Result<(), Error> {
    device.zero_grad()?;

    let _ = program.forward(&[&x])?;
    let loss = program.loss(y)?;

    let gradients = loss.backward()?;
    let gradients: &[Tensor] = &gradients.deref().borrow();

    optimizer.optimize(&gradients, learning_rate)?;

    Ok(())
}

pub fn time_it<F: Fn() -> T, T>(text: &str, f: F) -> T {
    let start = SystemTime::now();
    let result = f();
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    println!("{} took {} μs", text, duration.as_micros());
    result
}
