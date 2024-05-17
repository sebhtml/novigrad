mod gradient_descent;
pub use gradient_descent::*;

use crate::{Device, Error, Tensor};

pub trait OptimizerTrait {
    fn optimize(&self, device: &Device, tensors: &[Tensor]) -> Result<(), Error>;
}
