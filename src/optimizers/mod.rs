mod adam;
pub mod stochastic_gradient_descent;
pub use adam::*;
pub mod adam_w;
pub mod common_adam;

use crate::{tensor::Error, Device, Instruction, TensorWithGrad};

pub trait OptimizerTrait {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error>;
}
