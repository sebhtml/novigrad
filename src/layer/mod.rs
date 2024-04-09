mod linear;
pub use linear::*;
mod embedding;
pub use embedding::*;
mod reshape;
pub use reshape::*;

use crate::{DeltaWorkingMemory, Error, Sigmoid, SigmoidConfig, Softmax, SoftmaxConfig, Tensor};

pub trait Layer {
    fn plan_change(&mut self, previous_activation: &Tensor, layer_delta: &Tensor);

    fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error>;

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error>;

    // TODO backward should return Error
    fn backward(&self, layer_delta: &Tensor, previous_layer_delta: &mut Tensor);

    // TODO get_layer_delta should return Error
    fn get_layer_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        is_last_layer: bool,
        layer_delta: &mut Tensor,
    );
}

pub enum LayerConfig {
    Embedding(EmbeddingConfig),
    Linear(LinearConfig),
    Reshape(ReshapeConfig),
    Sigmoid(SigmoidConfig),
    Softmax(SoftmaxConfig),
}

pub enum LayerType {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

impl Into<LayerType> for &LayerConfig {
    fn into(self) -> LayerType {
        match self {
            LayerConfig::Embedding(config) => LayerType::Embedding(config.into()),
            LayerConfig::Linear(config) => LayerType::Linear(config.into()),
            LayerConfig::Reshape(config) => LayerType::Reshape(config.into()),
            LayerConfig::Sigmoid(config) => LayerType::Sigmoid(config.into()),
            LayerConfig::Softmax(config) => LayerType::Softmax(config.into()),
        }
    }
}

impl Layer for LayerType {
    fn plan_change(&mut self, previous_activation: &Tensor, layer_delta: &Tensor) {
        match self {
            LayerType::Embedding(that) => that.plan_change(previous_activation, layer_delta),
            LayerType::Linear(that) => that.plan_change(previous_activation, layer_delta),
            LayerType::Reshape(that) => that.plan_change(previous_activation, layer_delta),
            LayerType::Sigmoid(that) => that.plan_change(previous_activation, layer_delta),
            LayerType::Softmax(that) => that.plan_change(previous_activation, layer_delta),
        }
    }

    fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error> {
        match self {
            LayerType::Embedding(that) => that.commit_change(learning_rate),
            LayerType::Linear(that) => that.commit_change(learning_rate),
            LayerType::Reshape(that) => that.commit_change(learning_rate),
            LayerType::Sigmoid(that) => that.commit_change(learning_rate),
            LayerType::Softmax(that) => that.commit_change(learning_rate),
        }
    }

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        match self {
            LayerType::Embedding(that) => that.forward(input, output),
            LayerType::Linear(that) => that.forward(input, output),
            LayerType::Reshape(that) => that.forward(input, output),
            LayerType::Sigmoid(that) => that.forward(input, output),
            LayerType::Softmax(that) => that.forward(input, output),
        }
    }

    fn backward(&self, layer_delta: &Tensor, previous_layer_delta: &mut Tensor) {
        match self {
            LayerType::Embedding(that) => that.backward(layer_delta, previous_layer_delta),
            LayerType::Linear(that) => that.backward(layer_delta, previous_layer_delta),
            LayerType::Reshape(that) => that.backward(layer_delta, previous_layer_delta),
            LayerType::Sigmoid(that) => that.backward(layer_delta, previous_layer_delta),
            LayerType::Softmax(that) => that.backward(layer_delta, previous_layer_delta),
        }
    }

    fn get_layer_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        match self {
            LayerType::Embedding(that) => that.get_layer_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            LayerType::Linear(that) => that.get_layer_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            LayerType::Reshape(that) => that.get_layer_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            LayerType::Sigmoid(that) => that.get_layer_delta(
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            LayerType::Softmax(that) => that.get_layer_delta(
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
