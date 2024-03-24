use std::f32::consts::E;

use crate::{ActivationFunction, Tensor};

pub struct Softmax {}

impl Default for Softmax {
    fn default() -> Self {
        Self {}
    }
}

impl ActivationFunction for Softmax {
    fn activate(&self, mut product_matrix: Tensor) -> Tensor {
        let rows = product_matrix.rows();
        let cols = product_matrix.cols();
        let mut row = 0;
        while row < rows {
            // Find max

            let mut max = product_matrix.get(row, 0);
            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(row, col);
                max = max.max(x);
                col += 1;
            }

            // For each value:
            // 1. substract the max
            // 2. compute E^x
            // 3. add result to sum
            let mut sum = 0.0;
            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(row, col);
                let y = E.powf(x - max);
                product_matrix.set(row, col, y);
                sum += y;
                col += 1;
            }

            // Divide every value by sum.

            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(row, col);
                let y = x / sum;
                product_matrix.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
        product_matrix
    }

    fn derive(&self, mut matrix: Tensor) -> Tensor {
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
