use std::ops::Deref;

use crate::{Device, Error, LearningTensor, OptimizerTrait, TensorF32};

#[derive(Default)]
pub struct GradientDescent {}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, gradients: Vec<LearningTensor>, device: &Device) -> Result<(), Error> {
        let learning_rate: f32 = 0.5;
        for gradient in gradients {
            let tensor: &mut TensorF32 = &mut gradient.tensor().deref().borrow_mut();
            let gradient: &TensorF32 = &gradient.gradient().deref().borrow();
            debug_assert_eq!(gradient.shape(), tensor.shape(),);
            TensorF32::saxpy(device, -learning_rate, gradient, tensor)?;
        }
        Ok(())
    }
}
