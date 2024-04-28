#[cfg(test)]
pub mod tests;
mod train;
use std::{cell::RefCell, ops::Deref, rc::Rc, vec};
pub use train::*;

use crate::{
    devices::Device, Error, Forward, Operator, Optimizer, OptimizerTrait, Tape, Tensor, TensorF32,
};

pub struct Network {
    architecture: Box<dyn Forward>,
    loss_function: Operator,
    device: Rc<Device>,
    optimizer: Optimizer,
    tape: Rc<RefCell<Tape>>,
}

pub struct TrainWorkingMemory {
    pub layer_output: TensorF32,
    pub next_layer_delta: TensorF32,
    pub back_propagated_delta: TensorF32,
    pub layer_delta: TensorF32,
    pub previous_activation_tensor: TensorF32,
    pub tmp: TensorF32,
}

impl TrainWorkingMemory {
    pub fn new(device: &Device) -> Self {
        Self {
            layer_output: device.tensor(0, 0, vec![]),
            next_layer_delta: device.tensor(0, 0, vec![]),
            back_propagated_delta: device.tensor(0, 0, vec![]),
            layer_delta: device.tensor(0, 0, vec![]),
            previous_activation_tensor: device.tensor(0, 0, vec![]),
            tmp: device.tensor(0, 0, vec![]),
        }
    }
}

pub struct PredictWorkingMemory {
    pub previous_activation_tensor: TensorF32,
    pub activation_tensor: TensorF32,
    pub activation_tensors: Vec<TensorF32>,
}

impl PredictWorkingMemory {
    pub fn new(examples_count: usize, device: &Device) -> Self {
        let mut activation_tensors = vec![];
        for _ in 0..examples_count {
            activation_tensors.push(device.tensor(0, 0, vec![]))
        }
        Self {
            previous_activation_tensor: device.tensor(0, 0, vec![]),
            activation_tensor: device.tensor(0, 0, vec![]),
            activation_tensors,
        }
    }
}

impl Network {
    pub fn new(architecture: Box<dyn Forward>, loss_function: Operator) -> Self {
        let device = architecture.device();
        let tape = architecture.tape();
        Self {
            architecture,
            loss_function,
            device,
            tape,
            optimizer: Default::default(),
        }
    }

    pub fn train(
        &mut self,
        epoch: usize,
        inputs: &Vec<Tensor>,
        outputs: &Vec<Tensor>,
    ) -> Result<(), Error> {
        for i in 0..inputs.len() {
            self.train_back_propagation(epoch, i, &inputs[i], &outputs[i])?;
        }
        Ok(())
    }

    pub fn total_error(&mut self, inputs: &[Tensor], outputs: &[Tensor]) -> Result<f32, Error> {
        let mut total_error = 0.0;
        for i in 0..inputs.len() {
            let output = self.forward(&[inputs[i].clone()])?;
            let target = &outputs[i];
            let example_error = self
                .loss_function
                .forward(&[target.clone(), output.clone()])?;
            let example_error: &TensorF32 = &example_error.tensor().deref().borrow();
            let example_error: f32 = example_error.try_into()?;
            total_error += example_error;
        }

        Ok(total_error)
    }

    fn train_back_propagation(
        &mut self,
        _epoch: usize,
        _example_index: usize,
        x: &Tensor,
        y: &Tensor,
    ) -> Result<(), Error> {
        self.tape.deref().borrow_mut().clear();

        let output = self.forward(&[x.clone()])?;

        let loss = self.loss_function.forward(&[y.clone(), output.clone()])?;

        let gradients = loss.backward(&self.device, &self.tape)?;

        self.optimizer.optimize(gradients, &self.device)
    }

    pub fn predict_many(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let len = inputs.len();
        let mut outputs = vec![];
        let mut i = 0;
        while i < len {
            let input = &inputs[i];
            let output = self.forward(&[input.clone()])?;
            outputs.push(output);
            i += 1;
        }
        Ok(outputs)
    }
}

impl Forward for Network {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        self.architecture.forward(inputs)
    }

    fn device(&self) -> Rc<Device> {
        self.device.clone()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
    }
}
