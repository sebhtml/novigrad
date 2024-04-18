use crate::Error;
mod sigmoid;
pub use sigmoid::*;
mod softmax;
pub use softmax::*;

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
