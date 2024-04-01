mod linear;
use std::rc::Rc;

pub use linear::*;

use crate::{ActivationFunction, Error, Tensor};

pub trait Layer {
    fn weights<'a>(&'a self) -> &'a Tensor;
    fn apply_weight_deltas(
        &mut self,
        addition: &mut Tensor,
        weight_deltas: &Tensor,
    ) -> Result<(), Error>;
    fn activation(&self) -> Rc<dyn ActivationFunction>;
    fn forward(
        &self,
        input: &Tensor,
        matrix_product: &mut Tensor,
        activation_tensor: &mut Tensor,
    ) -> Result<(), Error>;
}
