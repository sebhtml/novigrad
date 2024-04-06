mod linear;
pub use linear::*;
mod embedding;
pub use embedding::*;
mod reshape;
pub use reshape::*;

use crate::{DeltaWorkingMemory, Error, Tensor};

pub trait Layer {
    fn plan_change(
        &mut self,
        learning_rate: f32,
        previous_activation: &Tensor,
        layer_delta: &Tensor,
    );

    fn commit_change(&mut self) -> Result<(), Error>;

    fn forward(&mut self, input: &Tensor) -> Result<(), Error>;

    fn get_activation_tensor<'a>(&'a self) -> &'a Tensor;

    // TODO backward should return Error
    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor);

    // TODO remove _using_softmax_and_cross_entropy_loss from trait
    // TODO get_layer_delta should return Error
    fn get_layer_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        next_layer: Option<&LayerType>,
        next_layer_delta: &Tensor,
        using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    );
}

pub enum LayerConfig {
    Embedding(EmbeddingConfig),
    Linear(LinearConfig),
    Reshape(ReshapeConfig),
}

pub enum LayerType {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
}

impl Into<LayerType> for &LayerConfig {
    fn into(self) -> LayerType {
        match self {
            LayerConfig::Embedding(config) => LayerType::Embedding(config.into()),
            LayerConfig::Linear(config) => LayerType::Linear(config.into()),
            LayerConfig::Reshape(config) => LayerType::Reshape(config.into()),
        }
    }
}

impl Layer for LayerType {
    fn plan_change(
        &mut self,
        learning_rate: f32,
        previous_activation: &Tensor,
        layer_delta: &Tensor,
    ) {
        match self {
            LayerType::Embedding(that) => {
                that.plan_change(learning_rate, previous_activation, layer_delta)
            }
            LayerType::Linear(that) => {
                that.plan_change(learning_rate, previous_activation, layer_delta)
            }
            LayerType::Reshape(that) => {
                that.plan_change(learning_rate, previous_activation, layer_delta)
            }
        }
    }

    fn commit_change(&mut self) -> Result<(), Error> {
        match self {
            LayerType::Embedding(that) => that.commit_change(),
            LayerType::Linear(that) => that.commit_change(),
            LayerType::Reshape(that) => that.commit_change(),
        }
    }

    fn forward(&mut self, input: &Tensor) -> Result<(), Error> {
        match self {
            LayerType::Embedding(that) => that.forward(input),
            LayerType::Linear(that) => that.forward(input),
            LayerType::Reshape(that) => that.forward(input),
        }
    }

    fn get_activation_tensor<'a>(&'a self) -> &'a Tensor {
        match self {
            LayerType::Embedding(that) => that.get_activation_tensor(),
            LayerType::Linear(that) => that.get_activation_tensor(),
            LayerType::Reshape(that) => that.get_activation_tensor(),
        }
    }

    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor) {
        match self {
            LayerType::Embedding(that) => that.backward(layer_delta, output_diff),
            LayerType::Linear(that) => that.backward(layer_delta, output_diff),
            LayerType::Reshape(that) => that.backward(layer_delta, output_diff),
        }
    }

    fn get_layer_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        next_layer: Option<&LayerType>,
        next_layer_delta: &Tensor,
        using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    ) {
        match self {
            LayerType::Embedding(that) => that.get_layer_delta(
                working_memory,
                next_layer,
                next_layer_delta,
                using_softmax_and_cross_entropy_loss,
                layer_delta,
            ),
            LayerType::Linear(that) => that.get_layer_delta(
                working_memory,
                next_layer,
                next_layer_delta,
                using_softmax_and_cross_entropy_loss,
                layer_delta,
            ),
            LayerType::Reshape(that) => that.get_layer_delta(
                working_memory,
                next_layer,
                next_layer_delta,
                using_softmax_and_cross_entropy_loss,
                layer_delta,
            ),
        }
    }
}
