mod sigmoid;
use std::rc::Rc;

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

impl Into<Rc<dyn ActivationFunction>> for Activation {
    fn into(self) -> Rc<dyn ActivationFunction> {
        match self {
            Activation::Sigmoid => Rc::new(Sigmoid::default()),
            Activation::Softmax => Rc::new(Softmax::default()),
        }
    }
}
