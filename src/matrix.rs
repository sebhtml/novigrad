use std::{
    fmt::Display,
    ops::{Add, Mul},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Self {
        Self { rows, cols, values }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.values[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.values[row * self.cols + col] = value;
    }

    pub fn transpose(&self) -> Self {
        let mut other = Matrix::new(self.cols, self.rows, self.values.clone());
        let mut row = 0;
        let rows = self.rows;
        while row < rows {
            let mut col = 0;
            let cols = self.cols;
            while col < cols {
                other.set(col, row, self.get(row, col));
                col += 1;
            }
            row += 1;
        }
        other
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    IncompatibleMatrixShapes,
}

impl Add for &Matrix {
    type Output = Result<Matrix, Error>;

    fn add(self, right: Self) -> Self::Output {
        let left = self;
        if left.rows != right.rows || left.cols != right.cols {
            return Err(Error::IncompatibleMatrixShapes);
        }

        let mut values = Vec::new();
        values.resize(left.values.len(), 0.0);

        let mut result = Matrix::new(left.rows, left.cols, values);
        let result_ptr = result.values.as_mut_ptr();
        let left_ptr = left.values.as_ptr();
        let right_ptr = right.values.as_ptr();

        unsafe {
            for index in 0..left.values.len() {
                let left_cell = left_ptr.add(index);
                let right_cell = right_ptr.add(index);
                let result_cell = result_ptr.add(index);
                *result_cell = *left_cell + *right_cell;
            }
        }

        Ok(result)
    }
}

// TODO for large matrices, this could be used:
// matmulImplLoopOrder algorithm
// from https://siboehm.com/articles/22/Fast-MMM-on-CPU
// from Simon Boehm who works at Anthropic
// Also see "matmulImplTiling" from this link.
impl Mul for &Matrix {
    type Output = Result<Matrix, Error>;

    fn mul(self, right: &Matrix) -> Self::Output {
        let left: &Matrix = self;
        if left.cols != right.rows {
            return Err(Error::IncompatibleMatrixShapes);
        }
        let mut result_values = Vec::new();
        result_values.resize(left.rows * right.cols, 0.0);
        let result_ptr = result_values.as_mut_ptr();
        let left_ptr = left.values.as_ptr();
        let right_ptr = right.values.as_ptr();

        unsafe {
            let mut row = 0;
            let left_rows = left.rows;
            let left_cols = left.cols;
            let right_cols = right.cols;
            while row != left_rows {
                let mut inner = 0;
                while inner != left_cols {
                    let mut col = 0;
                    while col != right_cols {
                        let left_cell = left_ptr.add(row * left.cols + inner);
                        let right_cell = right_ptr.add(inner * right.cols + col);
                        let result_cell = result_ptr.add(row * right.cols + col);
                        *result_cell += *left_cell * *right_cell;
                        col += 1;
                    }
                    inner += 1;
                }
                row += 1;
            }
        }

        let result = Matrix::new(left.rows, right.cols, result_values);
        Ok(result)
    }
}

impl Into<Vec<f32>> for Matrix {
    fn into(self) -> Vec<f32> {
        self.values
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        _ = write!(f, "Shape: {}x{}", self.rows, self.cols);
        _ = write!(f, "\n");
        for row in 0..self.rows {
            for col in 0..self.cols {
                _ = write!(f, " {:2.8}", self.values[row * self.cols + col]);
            }
            _ = write!(f, "\n");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::matrix::{Error, Matrix};

    #[test]
    fn new() {
        // Given rows and cols
        // When a matrix is built
        // Then it has the appropriate shape

        let matrix = Matrix::new(
            4,
            3,
            vec![
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
            ],
        );
        assert_eq!(matrix.shape(), (4, 3));
    }

    #[test]
    fn multiplication_shape_compatibility() {
        // Given two matrices with incompatible shapes
        // When a matrix multiplication is done
        // Then there is an error

        let lhs = Matrix::new(
            1,
            1,
            vec![
                0.0, //
            ],
        );

        let rhs = Matrix::new(
            2,
            1,
            vec![
                0.0, //
                0.0, //
            ],
        );
        let actual_product = &lhs * &rhs;
        assert_eq!(actual_product, Err(Error::IncompatibleMatrixShapes))
    }

    #[test]
    fn matrix_multiplication_result() {
        // Given a left-hand side matrix and and a right-hand side matrix
        // When the multiplication lhs * rhs is done
        // Then the resulting matrix has the correct values

        let lhs = Matrix::new(
            3,
            2,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        );
        let rhs = Matrix::new(
            2,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        );
        let actual_result = &lhs * &rhs;
        let expected_result = Matrix::new(
            3,
            3,
            vec![
                1.0 * 11.0 + 2.0 * 14.0,
                1.0 * 12.0 + 2.0 * 15.0,
                1.0 * 13.0 + 2.0 * 16.0, //
                3.0 * 11.0 + 4.0 * 14.0,
                3.0 * 12.0 + 4.0 * 15.0,
                3.0 * 13.0 + 4.0 * 16.0, //
                5.0 * 11.0 + 6.0 * 14.0,
                5.0 * 12.0 + 6.0 * 15.0,
                5.0 * 13.0 + 6.0 * 16.0, //
            ],
        );

        assert_eq!(actual_result, Ok(expected_result));
    }

    #[test]
    fn matrix_addition_result() {
        // Given a left-hand side matrix and and a right-hand side matrix
        // When the addition lhs + rhs is done
        // Then the resulting matrix has the correct values

        let lhs = Matrix::new(
            3,
            2,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        );
        let rhs = Matrix::new(
            3,
            2,
            vec![
                11.0, 12.0, //
                14.0, 15.0, //
                13.0, 16.0, //
            ],
        );
        let actual_result = &lhs + &rhs;
        let expected_result = Matrix::new(
            3,
            2,
            vec![
                1.0 + 11.0,
                2.0 + 12.0, //
                3.0 + 14.0,
                4.0 + 15.0, //
                5.0 + 13.0,
                6.0 + 16.0, //
            ],
        );

        assert_eq!(actual_result, Ok(expected_result));
    }

    #[test]
    fn big_matrix_multiplication() {
        let rows = 1024;
        let cols = 1024;
        let mut values = Vec::new();
        values.resize(rows * cols, 0.0);
        for index in 0..values.len() {
            values[index] = rand::thread_rng().gen_range(0.0..1.0)
        }
        let m = Matrix::new(rows, cols, values);
        let _result = &m * &m;
    }

    #[test]
    fn big_matrix_addition() {
        let rows = 1024;
        let cols = 1024;
        let mut values = Vec::new();
        values.resize(rows * cols, 0.0);
        for index in 0..values.len() {
            values[index] = rand::thread_rng().gen_range(0.0..1.0)
        }
        let m = Matrix::new(rows, cols, values);
        let _result = &m + &m;
    }

    #[test]
    fn transpose() {
        let matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix2 = matrix.transpose();
        for row in 0..matrix.rows() {
            for col in 0..matrix.cols() {
                assert_eq!(matrix2.get(col, row), matrix.get(row, col));
            }
        }
    }
}
