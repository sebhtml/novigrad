use std::ops::Deref;

use crate::{Device, Error, OptimizerTrait, Tensor, TensorF32};

pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, device: &Device, tensors: &[Tensor]) -> Result<(), Error> {
        for optimizable_tensor in tensors {
            let tensor: &TensorF32 = &optimizable_tensor.tensor().deref().borrow();
            let gradient: &TensorF32 = &optimizable_tensor.gradient().deref().borrow();
            debug_assert_eq!(gradient.size(), tensor.size(),);
            let tmp = device.tensor_f32(tensor.rows(), tensor.cols(), vec![0.0; tensor.len()]);
            TensorF32::scale(0.0, &tmp)?;
            TensorF32::add(gradient, &tmp)?;
            TensorF32::scale(-self.learning_rate, &tmp)?;
            TensorF32::add(&tmp, tensor)?;
        }
        Ok(())
    }
}
