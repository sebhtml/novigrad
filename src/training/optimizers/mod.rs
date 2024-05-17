mod gradient_descent;
pub use gradient_descent::*;

use crate::{Error, Tensor};

pub trait OptimizerTrait {
    fn optimize(&self, gradients: &[Tensor]) -> Result<(), Error>;
}
