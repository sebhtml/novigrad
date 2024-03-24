use crate::Error;
use crate::{ActivationFunction, Tensor};
use std::f32::consts::E;

pub struct Sigmoid {}

impl Default for Sigmoid {
    fn default() -> Self {
        Self {}
    }
}

impl ActivationFunction for Sigmoid {
    fn activate(&self, matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        result.reshape(matrix.rows(), matrix.cols());
        let rows = matrix.rows();
        let cols = matrix.cols();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = matrix.get(row, col);
                let y = 1.0 / (1.0 + E.powf(-x));
                result.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
        Ok(())
    }

    fn derive(&self, matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        result.reshape(matrix.rows(), matrix.cols());
        let rows = matrix.rows();
        let cols = matrix.cols();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = matrix.get(row, col);
                let y = x * (1.0 - x);
                result.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
        Ok(())
    }
}
