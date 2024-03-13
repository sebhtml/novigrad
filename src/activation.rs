use crate::Matrix;

pub trait ActivationFunction {
    fn activate_matrix(&self, matrix: Matrix) -> Matrix;

    fn derive_matrix(&self, matrix: Matrix) -> Matrix;
}
