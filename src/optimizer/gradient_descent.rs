use std::ops::Deref;

use crate::{Error, OptimizerTrait, Tensor, TensorF32};

#[derive(Default)]
pub struct GradientDescent {}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, gradients: &[Tensor], learning_rate: f32) -> Result<(), Error> {
        //println!("Optimizing {} model tensors", gradients.len());
        for gradient in gradients {
            let tensor: &mut TensorF32 = &mut gradient.tensor().deref().borrow_mut();
            let gradient: &TensorF32 = &gradient.gradient().deref().borrow();
            debug_assert_eq!(gradient.size(), tensor.size(),);
            TensorF32::a_x_plus_y(-learning_rate, gradient, tensor)?;
        }
        Ok(())
    }
}
