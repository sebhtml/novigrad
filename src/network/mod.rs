#[cfg(test)]
pub mod tests;
mod train;
use std::ops::Deref;
pub use train::*;

use crate::{devices::Device, Error, OperatorTrait, OptimizerTrait, Tensor, TensorF32};

pub struct Network {}

impl Network {
    pub fn train(
        model: &Box<dyn OperatorTrait>,
        loss_function: &Box<dyn OperatorTrait>,
        device: &Device,
        optimizer: &Box<dyn OptimizerTrait>,
        learning_rate: f32,
        epoch: usize,
        inputs: &Vec<Tensor>,
        outputs: &Vec<Tensor>,
    ) -> Result<(), Error> {
        for i in 0..inputs.len() {
            Self::train_back_propagation(
                model,
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
        let example_loss =
            loss_function.forward(&[expected_output.clone(), actual_output.clone()])?;
        let example_loss: &TensorF32 = &example_loss.tensor().deref().borrow();
        let example_loss: f32 = example_loss.try_into()?;
        Ok(example_loss)
    }

    pub fn total_loss(
        model: &Box<dyn OperatorTrait>,
        loss_function: &Box<dyn OperatorTrait>,
        inputs: &[Tensor],
        outputs: &[Tensor],
    ) -> Result<f32, Error> {
        let mut total_error = 0.0;
        for i in 0..inputs.len() {
            let output = model.forward(&[inputs[i].clone()])?;
            let expected_output = &outputs[i];
            let example_error = Self::example_loss(loss_function, &output, expected_output)?;
            total_error += example_error;
        }

        Ok(total_error)
    }

    fn train_back_propagation(
        model: &Box<dyn OperatorTrait>,
        loss_function: &Box<dyn OperatorTrait>,
        device: &Device,
        optimizer: &Box<dyn OptimizerTrait>,
        learning_rate: f32,
        _epoch: usize,
        _example_index: usize,
        x: &Tensor,
        y: &Tensor,
    ) -> Result<(), Error> {
        device.zero_grad()?;

        let output = model.forward(&[x.clone()])?;

        let loss = loss_function.forward(&[y.clone(), output.clone()])?;

        let gradients = loss.backward()?;
        let gradients: &[Tensor] = &gradients.deref().borrow();

        optimizer.optimize(&gradients, learning_rate)?;

        Ok(())
    }
}
