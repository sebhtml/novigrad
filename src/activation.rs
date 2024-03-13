use crate::{Matrix, Sigmoid, Softmax};

pub trait ActivationFunction {
    fn activate_matrix(&self, product_matrix: Matrix) -> Matrix;

    fn derive_matrix(&self, activation_matrix: Matrix) -> Matrix;
}

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
