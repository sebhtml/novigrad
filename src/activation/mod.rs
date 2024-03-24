mod sigmoid;
use crate::Error;
use sigmoid::*;
use std::rc::Rc;
mod softmax;
use softmax::*;

use crate::Tensor;

pub trait ActivationFunction {
    fn activate(&self, product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error>;
    fn derive(&self, activation_matrix: Tensor) -> Tensor;
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
