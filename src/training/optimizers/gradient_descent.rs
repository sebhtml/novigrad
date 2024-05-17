use std::ops::Deref;

use crate::{Error, OptimizerTrait, Tensor, TensorF32};

pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, gradients: &[Tensor]) -> Result<(), Error> {
        for gradient in gradients {
            let tensor: &mut TensorF32 = &mut gradient.tensor().deref().borrow_mut();
            let gradient: &TensorF32 = &gradient.gradient().deref().borrow();
            debug_assert_eq!(gradient.size(), tensor.size(),);
            TensorF32::a_x_plus_y(-self.learning_rate, gradient, tensor)?;
        }
        Ok(())
    }
}
