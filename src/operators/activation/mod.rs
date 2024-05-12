use crate::Error;
mod sigmoid;
pub use sigmoid::*;
mod softmax;
pub use softmax::*;

use crate::TensorF32;

pub trait ActivationFunction {
    fn activate(product_matrix: &TensorF32, result: &TensorF32) -> Result<(), Error>;
    fn derive(
        product_matrix: &TensorF32,
        activation_matrix: &TensorF32,
        result: &mut TensorF32,
    ) -> Result<(), Error>;
}
