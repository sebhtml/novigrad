mod gradient_descent;
pub use gradient_descent::*;
mod adam;
pub use adam::*;
pub mod common_adam;

use crate::{tensor::Error, Device, Instruction, TensorWithGrad};

pub trait OptimizerTrait {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error>;
}
