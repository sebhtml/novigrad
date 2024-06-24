mod model;
pub use model::*;
pub mod mega_man;
pub mod multi_head_attention_model;
pub mod perceptron;
pub mod simple;
pub mod transformer_model;
use crate::{BinaryOperator, Metrics, OptimizerTrait};

use crate::{Device, TensorWithGrad, Tokenizer};

pub struct ModelDetails<Model, LossOperator, Optimizer>
where
    Model: UnaryModel,
    LossOperator: BinaryOperator,
    Optimizer: OptimizerTrait,
{
    pub device: Device,
    pub tokenizer: Option<Tokenizer>,
    pub examples: Vec<(TensorWithGrad, TensorWithGrad)>,
    pub model: Model,
    pub loss_operator: LossOperator,
    pub optimizer: Optimizer,
    pub learning_rate: f32,
    pub shuffle_examples: bool,
    pub clipped_gradient_norm: bool,
    pub epochs: usize,
    pub progress: usize,
    pub initial_metrics: Metrics,
    pub final_metrics: Metrics,
    pub maximum_incorrect_argmaxes: usize,
}
