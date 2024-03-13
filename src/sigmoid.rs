use std::f32::consts::E;

use crate::{ActivationFunction, Matrix};

pub struct Sigmoid {}

impl Default for Sigmoid {
    fn default() -> Self {
        Self {}
    }
}

impl Sigmoid {
    fn activate(&self, x: f32) -> f32 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn derive(&self, x: f32) -> f32 {
        let sigmoid_x = self.activate(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }
}

impl ActivationFunction for Sigmoid {
    fn activate_matrix(&self, mut matrix: Matrix) -> Matrix {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let y = self.activate(matrix.get(row, col));
                matrix.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
        matrix
    }

    fn derive_matrix(&self, mut matrix: Matrix) -> Matrix {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let y = self.derive(matrix.get(row, col));
                matrix.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
        matrix
    }
}
