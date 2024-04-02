mod linear;
pub use linear::*;

use crate::{Activation, DeltaWorkingMemory, Error, Tensor};

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

    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor);

    fn get_layer_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        next_layer: Option<&Box<dyn Layer>>,
        next_layer_delta: &Tensor,
        using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    );
}

pub enum LayerType {
    Linear,
}

pub struct LayerConfig {
    pub layer_type: LayerType,
    pub input_rows: usize,
    pub rows: usize,
    pub cols: usize,
    pub activation: Activation,
}

impl Into<Box<dyn Layer>> for &LayerConfig {
    fn into(self) -> Box<dyn Layer> {
        Box::new(Linear::new(
            self.input_rows,
            self.rows,
            self.cols,
            self.activation.clone().into(),
        ))
    }
}
