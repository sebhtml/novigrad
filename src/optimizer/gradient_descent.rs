use crate::{Accelerator, Gradient, OptimizerTrait};

#[derive(Default)]
pub struct GradientDescent {}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, gradients: Vec<Gradient>, accelerator: &Accelerator) {
        let learning_rate: f32 = 0.5;
        for gradient in gradients {
            gradient.commit_change(accelerator, learning_rate);
        }
    }
}
