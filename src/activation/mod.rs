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
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

impl ActivationFunction for Activation {
    fn activate(&self, product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match self {
            Activation::Sigmoid(that) => that.activate(product_matrix, result),
            Activation::Softmax(that) => that.activate(product_matrix, result),
        }
    }

    fn derive(
        &self,
        product_matrix: &Tensor,
        activation_matrix: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            Activation::Sigmoid(that) => that.derive(product_matrix, activation_matrix, result),
            Activation::Softmax(that) => that.derive(product_matrix, activation_matrix, result),
        }
    }
}
