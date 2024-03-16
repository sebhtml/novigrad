use std::f32::consts::E;

use crate::{ActivationFunction, Tensor};

pub struct Softmax {}

impl Default for Softmax {
    fn default() -> Self {
        Self {}
    }
}

impl ActivationFunction for Softmax {
    fn activate_matrix(&self, mut product_matrix: Tensor) -> Tensor {
        // TODO generalize
        // Find max
        let rows = product_matrix.dimensions()[0];
        let cols = product_matrix.dimensions()[1];
        let mut max = product_matrix.get(&vec![0, 0]);
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(&vec![row, col]);
                max = max.max(x);
                col += 1;
            }
            row += 1;
        }
        // For each value:
        // 1. substract the max
        // 2. compute E^x
        // 3. add result to sum
        let mut sum = 0.0;
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(&vec![row, col]);
                let y = E.powf(x - max);
                product_matrix.set(&vec![row, col], y);
                sum += y;
                col += 1;
            }
            row += 1;
        }

        // Divide every value by sum.
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(&vec![row, col]);
                let y = x / sum;
                product_matrix.set(&vec![row, col], y);
                col += 1;
            }
            row += 1;
        }
        product_matrix
    }

    fn derive_matrix(&self, mut matrix: Tensor) -> Tensor {
        // TODO generalize
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
