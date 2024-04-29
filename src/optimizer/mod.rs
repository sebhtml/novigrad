mod gradient_descent;
pub use gradient_descent::*;

use crate::{Device, Error, Tensor};

pub trait OptimizerTrait {
    fn optimize(
        &self,
        gradients: &[Tensor],
        device: &Device,
        learning_rate: f32,
    ) -> Result<(), Error>;
}

pub enum Optimizer {
    GradientDescent(GradientDescent),
}

impl OptimizerTrait for Optimizer {
    fn optimize(
        &self,
        gradients: &[Tensor],
        device: &Device,
        learning_rate: f32,
    ) -> Result<(), Error> {
        match self {
            Optimizer::GradientDescent(object) => object.optimize(gradients, device, learning_rate),
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::GradientDescent(GradientDescent::default())
    }
}
