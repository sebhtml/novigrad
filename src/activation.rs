use crate::Matrix;

pub trait ActivationFunction {
    fn activate_matrix(&self, product_matrix: Matrix) -> Matrix;

    fn derive_matrix(&self, activation_matrix: Matrix) -> Matrix;
}
