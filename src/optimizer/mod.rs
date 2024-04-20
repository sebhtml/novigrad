mod gradient_descent;
use std::{cell::RefCell, rc::Rc};

pub use gradient_descent::*;

use crate::{Accelerator, Gradient, Tape};

pub trait OptimizerTrait {
    fn optimize(
        &self,
        tape: &Rc<RefCell<Tape>>,
        gradients: Vec<Gradient>,
        accelerator: &Accelerator,
    );
}

pub enum Optimizer {
    GradientDescent(GradientDescent),
}

impl OptimizerTrait for Optimizer {
    fn optimize(
        &self,
        tape: &Rc<RefCell<Tape>>,
        gradients: Vec<Gradient>,
        accelerator: &Accelerator,
    ) {
        match self {
            Optimizer::GradientDescent(object) => object.optimize(tape, gradients, accelerator),
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::GradientDescent(Default::default())
    }
}
