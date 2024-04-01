mod linear;
pub use linear::*;

use crate::{DeltaWorkingMemory, Error, Tensor};

pub trait Layer {
    fn plan_change(
        &mut self,
        learning_rate: f32,
        previous_activation: &Tensor,
        layer_delta: &Tensor,
    );

    fn commit_change(&mut self) -> Result<(), Error>;

    fn forward(
        &self,
        input: &Tensor,
        matrix_product: &mut Tensor,
        activation_tensor: &mut Tensor,
    ) -> Result<(), Error>;

    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor);

    fn get_layer_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        layer_product_tensor: &Tensor,
        layer_activation_tensor: &Tensor,
        next_layer: Option<&Box<dyn Layer>>,
        next_layer_delta: &Tensor,
        using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    );
}
