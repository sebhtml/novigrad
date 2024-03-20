use std::f32::consts::E;

use crate::{ActivationFunction, Tensor};

pub struct Sigmoid {}

impl Default for Sigmoid {
    fn default() -> Self {
        Self {}
    }
}

impl ActivationFunction for Sigmoid {
    fn activate_matrix(&self, mut matrix: Tensor) -> Tensor {
        let rows = matrix.dimensions()[0];
        let cols = matrix.dimensions()[1];
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = matrix.get(&vec![row, col]);
                let y = 1.0 / (1.0 + E.powf(-x));
                matrix.set(&vec![row, col], y);
                col += 1;
            }
            row += 1;
        }
        matrix
    }

    fn derive_matrix(&self, mut matrix: Tensor) -> Tensor {
        let rows = matrix.dimensions()[0];
        let cols = matrix.dimensions()[1];
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = matrix.get(&vec![row, col]);
                let y = x * (1.0 - x);
                matrix.set(&vec![row, col], y);
                col += 1;
            }
            row += 1;
        }
        matrix
    }
}
