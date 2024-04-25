use std::ops::Deref;

use crate::{Device, LearningTensor, OptimizerTrait, Tensor};

#[derive(Default)]
pub struct GradientDescent {}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, gradients: Vec<LearningTensor>, device: &Device) {
        let learning_rate: f32 = 0.5;
        for gradient in gradients {
            let tensor: &mut Tensor = &mut gradient.tensor().deref().borrow_mut();
            let gradient: &Tensor = &gradient.gradient().deref().borrow();

            let op_result = Tensor::saxpy(device, -learning_rate, gradient, tensor);
            op_result.expect("Ok");
        }
    }
}
