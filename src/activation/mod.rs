mod sigmoid;
use crate::Error;
use sigmoid::*;
mod softmax;
use softmax::*;

use crate::Tensor;

pub trait ActivationFunction {
    fn activate(&self, product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error>;
    fn derive(
        &self,
        product_matrix: &Tensor,
        activation_matrix: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error>;
}

#[derive(Clone, PartialEq)]
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
