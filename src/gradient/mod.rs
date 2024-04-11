use std::mem::swap;

use crate::{
    DeltaWorkingMemory, Embedding, EmbeddingConfig, Error, Linear, LinearConfig, Reshape,
    ReshapeConfig, Sigmoid, SigmoidConfig, Softmax, SoftmaxConfig, Tensor,
};

pub struct DifferentiableTensor {
    pub tensor: Tensor,
    pub gradient: Tensor,
    pub has_gradient: bool,
    tmp: Tensor,
    addition: Tensor,
}

impl DifferentiableTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            gradient: Default::default(),
            has_gradient: Default::default(),
            tmp: Default::default(),
            addition: Default::default(),
        }
    }
    pub fn commit_change(&mut self, learning_rate: f32) {
        if !self.has_gradient {
            return;
        }
        let tmp = &mut self.tmp;
        let addition = &mut self.addition;
        // TODO use gemm C = tensor * identity + -learning_rate * gradient
        let op_result = self.gradient.scalar_mul(-learning_rate, tmp);
        op_result.expect("Ok");
        let op_result = self.tensor.add(&tmp, addition);
        op_result.expect("Ok");
        swap(&mut self.tensor, addition);
        self.has_gradient = false;
    }
}

impl From<Tensor> for DifferentiableTensor {
    fn from(value: Tensor) -> Self {
        Self::new(value)
    }
}

pub trait DifferentiableModuleTrait {
    fn compute_gradient(&mut self, layer_input: &Tensor, layer_output_delta: &Tensor);

    fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error>;

    fn forward(&mut self, layer_input: &Tensor, layer_output: &mut Tensor) -> Result<(), Error>;

    // TODO backward should return Error
    fn backward(&self, layer_output_delta: &Tensor, previous_layer_output_delta: &mut Tensor);

    // TODO get_layer_delta should return Error
    fn get_layer_output_delta(
        &self,
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

pub enum DifferentiableModule {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

impl Into<DifferentiableModule> for &DifferentiableModuleConfig {
    fn into(self) -> DifferentiableModule {
        match self {
            DifferentiableModuleConfig::Embedding(config) => {
                DifferentiableModule::Embedding(config.into())
            }
            DifferentiableModuleConfig::Linear(config) => {
                DifferentiableModule::Linear(config.into())
            }
            DifferentiableModuleConfig::Reshape(config) => {
                DifferentiableModule::Reshape(config.into())
            }
            DifferentiableModuleConfig::Sigmoid(config) => {
                DifferentiableModule::Sigmoid(config.into())
            }
            DifferentiableModuleConfig::Softmax(config) => {
                DifferentiableModule::Softmax(config.into())
            }
        }
    }
}

impl DifferentiableModuleTrait for DifferentiableModule {
    fn compute_gradient(&mut self, layer_input: &Tensor, layer_output_delta: &Tensor) {
        match self {
            DifferentiableModule::Embedding(that) => {
                that.compute_gradient(layer_input, layer_output_delta)
            }
            DifferentiableModule::Linear(that) => {
                that.compute_gradient(layer_input, layer_output_delta)
            }
            DifferentiableModule::Reshape(that) => {
                that.compute_gradient(layer_input, layer_output_delta)
            }
            DifferentiableModule::Sigmoid(that) => {
                that.compute_gradient(layer_input, layer_output_delta)
            }
            DifferentiableModule::Softmax(that) => {
                that.compute_gradient(layer_input, layer_output_delta)
            }
        }
    }

    fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error> {
        match self {
            DifferentiableModule::Embedding(that) => that.commit_change(learning_rate),
            DifferentiableModule::Linear(that) => that.commit_change(learning_rate),
            DifferentiableModule::Reshape(that) => that.commit_change(learning_rate),
            DifferentiableModule::Sigmoid(that) => that.commit_change(learning_rate),
            DifferentiableModule::Softmax(that) => that.commit_change(learning_rate),
        }
    }

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        match self {
            DifferentiableModule::Embedding(that) => that.forward(input, output),
            DifferentiableModule::Linear(that) => that.forward(input, output),
            DifferentiableModule::Reshape(that) => that.forward(input, output),
            DifferentiableModule::Sigmoid(that) => that.forward(input, output),
            DifferentiableModule::Softmax(that) => that.forward(input, output),
        }
    }

    fn backward(&self, layer_delta: &Tensor, previous_layer_delta: &mut Tensor) {
        match self {
            DifferentiableModule::Embedding(that) => {
                that.backward(layer_delta, previous_layer_delta)
            }
            DifferentiableModule::Linear(that) => that.backward(layer_delta, previous_layer_delta),
            DifferentiableModule::Reshape(that) => that.backward(layer_delta, previous_layer_delta),
            DifferentiableModule::Sigmoid(that) => that.backward(layer_delta, previous_layer_delta),
            DifferentiableModule::Softmax(that) => that.backward(layer_delta, previous_layer_delta),
        }
    }

    fn get_layer_output_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        match self {
            DifferentiableModule::Embedding(that) => that.get_layer_output_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Linear(that) => that.get_layer_output_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Reshape(that) => that.get_layer_output_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Sigmoid(that) => that.get_layer_output_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Softmax(that) => that.get_layer_output_delta(
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
