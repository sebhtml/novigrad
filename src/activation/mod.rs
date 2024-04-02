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
pub enum ActivationType {
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

impl ActivationFunction for ActivationType {
    fn activate(&self, product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match self {
            ActivationType::Sigmoid(that) => that.activate(product_matrix, result),
            ActivationType::Softmax(that) => that.activate(product_matrix, result),
        }
    }

    fn derive(
        &self,
        product_matrix: &Tensor,
        activation_matrix: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            ActivationType::Sigmoid(that) => that.derive(product_matrix, activation_matrix, result),
            ActivationType::Softmax(that) => that.derive(product_matrix, activation_matrix, result),
        }
    }
}
