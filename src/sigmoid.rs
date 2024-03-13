use std::f32::consts::E;

use crate::{ActivationFunction, Matrix};

pub struct Sigmoid {}

impl Default for Sigmoid {
    fn default() -> Self {
        Self {}
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
                let x = matrix.get(row, col);
                let y = 1.0 / (1.0 + E.powf(-x));
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
                let x = matrix.get(row, col);
                let y = x * (1.0 - x);
                matrix.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
        matrix
    }
}
