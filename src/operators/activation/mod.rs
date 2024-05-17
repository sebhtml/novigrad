use crate::Error;
mod sigmoid;
pub use sigmoid::*;
mod softmax;
pub use softmax::*;

use crate::TensorF32;

pub trait ActivationFunction {
    fn activate(input: &TensorF32, output: &TensorF32) -> Result<(), Error>;
    fn derive(
        input: &TensorF32,
        activation_output: &TensorF32,
        output: &mut TensorF32,
    ) -> Result<(), Error>;
}
