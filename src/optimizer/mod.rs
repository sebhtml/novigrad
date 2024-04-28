mod gradient_descent;
pub use gradient_descent::*;

use crate::{Device, Error, Tensor};

pub trait OptimizerTrait {
    fn optimize(&self, gradients: Vec<Tensor>, device: &Device) -> Result<(), Error>;
}

pub enum Optimizer {
    GradientDescent(GradientDescent),
}

impl OptimizerTrait for Optimizer {
    fn optimize(&self, gradients: Vec<Tensor>, device: &Device) -> Result<(), Error> {
        match self {
            Optimizer::GradientDescent(object) => object.optimize(gradients, device),
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::GradientDescent(GradientDescent::default())
    }
}
