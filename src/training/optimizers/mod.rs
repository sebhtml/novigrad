mod gradient_descent;
pub use gradient_descent::*;

use crate::{Device, Error, Instruction, TensorWithGrad};

pub trait OptimizerTrait {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error>;
}
