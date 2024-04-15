use std::{borrow::Borrow, cell::RefCell, ops::Deref, rc::Rc};

use crate::{
    accelerator::Accelerator, DeltaWorkingMemory, Embedding, EmbeddingConfig, Error, Linear,
    LinearConfig, Reshape, ReshapeConfig, Sigmoid, SigmoidConfig, Softmax, SoftmaxConfig, Tensor,
};

mod tape;
pub use tape::*;

pub struct DifferentiableTensor {
    pub tensor: Tensor,
    pub gradient: Tensor,
    pub has_gradient: bool,
}

impl DifferentiableTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            gradient: Default::default(),
            has_gradient: Default::default(),
        }
    }
    pub fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32) {
        if !self.has_gradient {
            return;
        }

        let op_result = Tensor::saxpy(
            accelerator,
            -learning_rate,
            &self.gradient,
            &mut self.tensor,
        );
        op_result.expect("Ok");
        self.has_gradient = false;
    }
}

impl From<Tensor> for DifferentiableTensor {
    fn from(value: Tensor) -> Self {
        Self::new(value)
    }
}

pub trait DifferentiableModuleTrait {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    );

    fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32)
        -> Result<(), Error>;

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output: &mut Tensor,
    ) -> Result<(), Error>;

    // TODO backward should return Error
    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    );

    // TODO get_layer_delta should return Error
    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        is_last_layer: bool,
        layer_output_delta: &mut Tensor,
    );
}

pub enum DifferentiableModuleConfig {
    Embedding(EmbeddingConfig),
    Linear(LinearConfig),
    Reshape(ReshapeConfig),
    Sigmoid(SigmoidConfig),
    Softmax(SoftmaxConfig),
}

pub enum DifferentiableModuleEnum {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

pub struct DifferentiableModule {
    tape: Rc<RefCell<Tape>>,
    variant: Rc<RefCell<DifferentiableModuleEnum>>,
}

pub struct FullDifferentiableModuleConfig<'a> {
    pub tape: &'a Rc<RefCell<Tape>>,
    pub config: &'a DifferentiableModuleConfig,
}

impl<'a> Into<DifferentiableModule> for &FullDifferentiableModuleConfig<'a> {
    fn into(self) -> DifferentiableModule {
        let config = self.config;
        let variant = match config {
            DifferentiableModuleConfig::Embedding(config) => {
                DifferentiableModuleEnum::Embedding(config.into())
            }
            DifferentiableModuleConfig::Linear(config) => {
                DifferentiableModuleEnum::Linear(config.into())
            }
            DifferentiableModuleConfig::Reshape(config) => {
                DifferentiableModuleEnum::Reshape(config.into())
            }
            DifferentiableModuleConfig::Sigmoid(config) => {
                DifferentiableModuleEnum::Sigmoid(config.into())
            }
            DifferentiableModuleConfig::Softmax(config) => {
                DifferentiableModuleEnum::Softmax(config.into())
            }
        };
        let tape = self.tape;
        DifferentiableModule {
            tape: tape.clone(),
            variant: Rc::new(RefCell::new(variant)),
        }
    }
}

impl DifferentiableModuleTrait for DifferentiableModule {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        match variant {
            DifferentiableModuleEnum::Embedding(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Linear(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Reshape(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Sigmoid(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Softmax(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
        }
    }

    fn commit_change(
        &mut self,
        accelerator: &Accelerator,
        learning_rate: f32,
    ) -> Result<(), Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        match variant {
            DifferentiableModuleEnum::Embedding(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Linear(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Reshape(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Sigmoid(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Softmax(that) => {
                that.commit_change(accelerator, learning_rate)
            }
        }
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        match variant {
            DifferentiableModuleEnum::Embedding(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Linear(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Reshape(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Sigmoid(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Softmax(that) => that.forward(accelerator, input, output),
        }?;
        self.tape
            .deref()
            .borrow_mut()
            .push(self.variant.borrow(), Rc::new(output.clone()).borrow());
        Ok(())
    }

    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_delta: &Tensor,
        previous_layer_delta: &mut Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        match variant {
            DifferentiableModuleEnum::Embedding(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Linear(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Reshape(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Sigmoid(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Softmax(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
        }
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        match variant {
            DifferentiableModuleEnum::Embedding(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Linear(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Reshape(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Sigmoid(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Softmax(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
        }
    }
}
