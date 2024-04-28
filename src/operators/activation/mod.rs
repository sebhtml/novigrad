use crate::Error;
mod sigmoid;
pub use sigmoid::*;
mod softmax;
pub use softmax::*;

use crate::TensorF32;

pub trait ActivationFunction {
    fn activate(&self, product_matrix: &TensorF32, result: &mut TensorF32) -> Result<(), Error>;
    fn derive(
        &self,
        product_matrix: &TensorF32,
        activation_matrix: &TensorF32,
        result: &mut TensorF32,
    ) -> Result<(), Error>;
}
