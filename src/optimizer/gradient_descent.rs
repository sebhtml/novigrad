use std::{cell::RefCell, rc::Rc};

use crate::{Accelerator, Gradient, OptimizerTrait, Tape};

#[derive(Default)]
pub struct GradientDescent {}

impl OptimizerTrait for GradientDescent {
    fn optimize(
        &self,
        _tape: &Rc<RefCell<Tape>>,
        gradients: Vec<Gradient>,
        accelerator: &Accelerator,
    ) {
        let learning_rate: f32 = 0.5;
        for gradient in gradients {
            gradient.commit_change(accelerator, learning_rate);
        }
    }
}
