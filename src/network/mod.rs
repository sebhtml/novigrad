#[cfg(test)]
pub mod tests;
mod train;
use std::ops::Deref;
pub use train::*;

use crate::{devices::Device, Error, OperatorTrait, Optimizer, OptimizerTrait, Tensor, TensorF32};

pub struct Network {
    model: Box<dyn OperatorTrait>,
    loss_function: Box<dyn OperatorTrait>,
    device: Device,
    optimizer: Optimizer,
}

impl Network {
    pub fn new(
        model: Box<dyn OperatorTrait>,
        loss_function: Box<dyn OperatorTrait>,
        device: &Device,
    ) -> Self {
        Self {
            model,
            loss_function,
            device: device.clone(),
            optimizer: Default::default(),
        }
    }

    pub fn train(
        &mut self,
        learning_rate: f32,
        epoch: usize,
        inputs: &Vec<Tensor>,
        outputs: &Vec<Tensor>,
    ) -> Result<(), Error> {
        for i in 0..inputs.len() {
            self.train_back_propagation(learning_rate, epoch, i, &inputs[i], &outputs[i])?;
        }
        Ok(())
    }

    pub fn example_loss(
        &self,
        actual_output: &Tensor,
        expected_output: &Tensor,
    ) -> Result<f32, Error> {
        let example_loss = self
            .loss_function
            .forward(&[expected_output.clone(), actual_output.clone()])?;
        let example_loss: &TensorF32 = &example_loss.tensor().deref().borrow();
        let example_loss: f32 = example_loss.try_into()?;
        Ok(example_loss)
    }

    pub fn total_loss(&self, inputs: &[Tensor], outputs: &[Tensor]) -> Result<f32, Error> {
        let mut total_error = 0.0;
        for i in 0..inputs.len() {
            let output = self.forward(&[inputs[i].clone()])?;
            let expected_output = &outputs[i];
            let example_error = self.example_loss(&output, expected_output)?;
            total_error += example_error;
        }

        Ok(total_error)
    }

    fn train_back_propagation(
        &mut self,
        learning_rate: f32,
        _epoch: usize,
        _example_index: usize,
        x: &Tensor,
        y: &Tensor,
    ) -> Result<(), Error> {
        self.device.zero_grad()?;

        let output = self.forward(&[x.clone()])?;

        let loss = self.loss_function.forward(&[y.clone(), output.clone()])?;

        let gradients = loss.backward()?;
        let gradients: &[Tensor] = &gradients.deref().borrow();

        self.optimizer.optimize(&gradients, learning_rate)?;

        Ok(())
    }
}

impl OperatorTrait for Network {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        //println!("---BEGIN---");
        let output = self.model.as_ref().forward(inputs);
        //println!("---END---");
        output
    }

    fn name(&self) -> &str {
        "Network"
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::UnsupportedOperation)
    }
}
