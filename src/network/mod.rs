#[cfg(test)]
pub mod tests;
mod train;
use std::{cell::RefCell, ops::Deref, rc::Rc, vec};
pub use train::*;

use crate::{
    devices::Device, Error, Forward, LearningTensor, Operator, Optimizer, OptimizerTrait, Tape,
    TensorF32,
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

pub struct DeltaWorkingMemory {
    pub layer_f_derivative: TensorF32,
}

impl DeltaWorkingMemory {
    pub fn new(device: &Device) -> Self {
        Self {
            layer_f_derivative: device.tensor(0, 0, vec![]),
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
        error_working_memory: &mut DeltaWorkingMemory,
        epoch: usize,
        inputs: &Vec<LearningTensor>,
        outputs: &Vec<LearningTensor>,
    ) -> Result<(), Error> {
        for i in 0..inputs.len() {
            self.train_back_propagation(error_working_memory, epoch, i, &inputs[i], &outputs[i])?;
        }
        Ok(())
    }

    pub fn total_error(
        &mut self,
        inputs: &[LearningTensor],
        outputs: &[LearningTensor],
    ) -> Result<f32, Error> {
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
        error_working_memory: &mut DeltaWorkingMemory,
        _epoch: usize,
        _example_index: usize,
        x: &LearningTensor,
        y: &LearningTensor,
    ) -> Result<(), Error> {
        self.tape.deref().borrow_mut().clear();

        let output = self.forward(&[x.clone()])?;

        let loss = self.loss_function.forward(&[y.clone(), output.clone()])?;

        let gradients = loss.backward(error_working_memory, &self.device, &self.tape)?;

        self.optimizer.optimize(gradients, &self.device)
    }

    pub fn predict_many(
        &mut self,
        inputs: &[LearningTensor],
    ) -> Result<Vec<LearningTensor>, Error> {
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
    fn forward(&self, inputs: &[LearningTensor]) -> Result<LearningTensor, Error> {
        self.architecture.forward(inputs)
    }

    fn device(&self) -> Rc<Device> {
        self.device.clone()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
    }
}
