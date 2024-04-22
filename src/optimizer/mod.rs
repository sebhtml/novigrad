mod gradient_descent;
pub use gradient_descent::*;

use crate::{Device, Gradient};

pub trait OptimizerTrait {
    fn optimize(&self, gradients: Vec<Gradient>, accelerator: &Device);
}

pub enum Optimizer {
    GradientDescent(GradientDescent),
}

impl OptimizerTrait for Optimizer {
    fn optimize(&self, gradients: Vec<Gradient>, accelerator: &Device) {
        match self {
            Optimizer::GradientDescent(object) => object.optimize(gradients, accelerator),
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::GradientDescent(GradientDescent::default())
    }
}
