mod gradient_descent;
use std::{cell::RefCell, rc::Rc};

pub use gradient_descent::*;

use crate::{Accelerator, Tape};

pub trait OptimizerTrait {
    fn optimize(&self, tape: &Rc<RefCell<Tape>>, accelerator: &Accelerator);
}

pub enum Optimizer {
    GradientDescent(GradientDescent),
}

impl OptimizerTrait for Optimizer {
    fn optimize(&self, tape: &Rc<RefCell<Tape>>, accelerator: &Accelerator) {
        match self {
            Optimizer::GradientDescent(object) => object.optimize(tape, accelerator),
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::GradientDescent(Default::default())
    }
}
