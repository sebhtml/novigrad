#[cfg(test)]
pub mod tests;
mod train;
use std::{ops::Deref, time::SystemTime};

pub use train::*;
mod learning_tensor;
use crate::{devices::Device, Error, Model, OperatorTrait, OptimizerTrait, Program, TensorF32};
pub use learning_tensor::*;

pub fn train(
    model: &Box<dyn Model>,
    program: &Program,
    loss_function: &Box<dyn OperatorTrait>,
    device: &Device,
    optimizer: &Box<dyn OptimizerTrait>,
    learning_rate: f32,
    epoch: usize,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
) -> Result<(), Error> {
    for i in 0..inputs.len() {
        train_back_propagation(
            model,
            program,
            loss_function,
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

pub fn example_loss(
    loss_function: &Box<dyn OperatorTrait>,
    actual_output: &Tensor,
    expected_output: &Tensor,
) -> Result<f32, Error> {
    let example_loss = loss_function.forward(&[&expected_output, &actual_output])?;
    let tape = example_loss.get_tape();
    for o in tape.iter() {
        o.realize()?;
    }

    let example_loss: &TensorF32 = &example_loss.tensor().deref().borrow();
    let example_loss: f32 = example_loss.try_into()?;
    Ok(example_loss)
}

pub fn total_loss(
    model: &Box<dyn Model>,
    loss_function: &Box<dyn OperatorTrait>,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> Result<f32, Error> {
    let mut total_error = 0.0;
    for i in 0..inputs.len() {
        let output = model.forward(&[&inputs[i]])?;
        let expected_output = &outputs[i];
        let example_error = example_loss(loss_function, &output, expected_output)?;
        total_error += example_error;
    }

    Ok(total_error)
}

fn train_back_propagation(
    model: &Box<dyn Model>,
    program: &Program,
    loss_function: &Box<dyn OperatorTrait>,
    device: &Device,
    optimizer: &Box<dyn OptimizerTrait>,
    learning_rate: f32,
    epoch: usize,
    example_index: usize,
    x: &Tensor,
    y: &Tensor,
) -> Result<(), Error> {
    device.zero_grad()?;

    let model_output = model.forward(&[&x])?;
    let model_loss = loss_function.forward(&[&y, &model_output])?;
    let tape = model_loss.get_tape();
    for o in tape.iter() {
        o.realize()?;
    }

    let program_output = program.forward(&[&x])?;
    let program_loss = loss_function.forward(&[&y, &model_output])?;
    let tape = program_loss.get_tape();
    for o in tape.iter() {
        o.realize()?;
    }

    let loss = program_loss.clone();
    let tape = loss.get_tape();
    for o in tape.iter() {
        o.realize()?;
    }

    {
        if &model_loss != &program_loss {
            println!("PANIC epoch {}  example {}", epoch, example_index);

            println!("x {}", x);
            let tape1 = model_output.get_tape();
            let tape2 = program_output.get_tape();
            println!("output tape {}", tape1.len());
            println!("program_output tape {}", tape2.len());
            let input1 = &tape1[0].inputs()[0];
            let input2 = &tape2[0].inputs()[0];
            println!("input1 {}", input1);
            println!("input2 {}", input2);
            for i in 0..tape1.len() {
                assert_eq!(tape1[i].operator().name(), tape2[i].operator().name());
                println!("{} {}", i, tape1[i].operator().name());
                println!("output {}", tape1[i]);
                println!("program_output {}", tape2[i]);
                assert_eq!(tape1[i], tape2[i]);
            }

            panic!();
        }
    }

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
