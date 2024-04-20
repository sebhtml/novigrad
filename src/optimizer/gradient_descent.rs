use std::ops::Deref;

use crate::{Accelerator, Gradient, OptimizerTrait, Tensor};

#[derive(Default)]
pub struct GradientDescent {}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, gradients: Vec<Gradient>, accelerator: &Accelerator) {
        let learning_rate: f32 = 0.5;
        for gradient in gradients {
            let tensor: &mut Tensor = &mut gradient.tensor().deref().borrow_mut();
            let gradient = gradient.gradient();
            let op_result = Tensor::saxpy(accelerator, -learning_rate, gradient, tensor);
            op_result.expect("Ok");
        }
    }
}
