mod sigmoid;
use sigmoid::*;
mod softmax;
use softmax::*;

use crate::Tensor;

pub trait ActivationFunction {
    fn activate_matrix(&self, product_matrix: Tensor) -> Tensor;

    fn derive_matrix(&self, activation_matrix: Tensor) -> Tensor;
}

#[derive(Clone)]
pub enum Activation {
    Sigmoid,
    Softmax,
}

impl Into<Box<dyn ActivationFunction>> for Activation {
    fn into(self) -> Box<dyn ActivationFunction> {
        match self {
            Activation::Sigmoid => Box::new(Sigmoid::default()),
            Activation::Softmax => Box::new(Softmax::default()),
        }
    }
}
