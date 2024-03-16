use std::{
    fmt::Display,
    ops::{Add, Mul},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    dimensions: Vec<usize>,
    values: Vec<f32>,
}

impl Tensor {
    pub fn new(dimensions: Vec<usize>, values: Vec<f32>) -> Self {
        Self { dimensions, values }
    }

    pub fn dimensions(&self) -> Vec<usize> {
        self.dimensions.clone()
    }

    pub fn index(&self, indices: &Vec<usize>) -> usize {
        // TODO generalize
        let index = indices[0] * self.dimensions[1] + indices[1];
        index
    }

    pub fn get(&self, indices: &Vec<usize>) -> f32 {
        let index = self.index(indices);
        self.values[index]
    }

    pub fn set(&mut self, indices: &Vec<usize>, value: f32) {
        let index = self.index(indices);
        self.values[index] = value;
    }

    pub fn transpose(&self) -> Self {
        // TODO generalize
        let rev_dimensions = self.dimensions.clone().into_iter().rev().collect();
        let mut other: Tensor = Tensor::new(rev_dimensions, self.values.clone());
        let mut row = 0;
        let rows = self.dimensions[0];
        let cols = self.dimensions[1];
        while row < rows {
            let mut col = 0;
            while col < cols {
                let value = self.get(&vec![row, col]);
                other.set(&vec![col, row], value);
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

impl Add for &Tensor {
    type Output = Result<Tensor, Error>;

    fn add(self, right: Self) -> Self::Output {
        let left = self;
        if left.dimensions != right.dimensions {
            return Err(Error::IncompatibleMatrixShapes);
        }

        let mut values = Vec::new();
        values.resize(left.values.len(), 0.0);

        let mut result = Tensor::new(left.dimensions.clone(), values);
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

// for large matrices, this could be used:
// matmulImplLoopOrder algorithm
// from https://siboehm.com/articles/22/Fast-MMM-on-CPU
// from Simon Boehm who works at Anthropic
// Also see "matmulImplTiling" from this link.
impl Mul for &Tensor {
    type Output = Result<Tensor, Error>;

    fn mul(self, right: &Tensor) -> Self::Output {
        let left: &Tensor = self;
        // TODO generalize
        if left.dimensions[1] != right.dimensions[0] {
            return Err(Error::IncompatibleMatrixShapes);
        }
        let mut result_values = Vec::new();
        result_values.resize(left.dimensions[0] * right.dimensions[1], 0.0);
        let result_ptr = result_values.as_mut_ptr();
        let left_ptr = left.values.as_ptr();
        let right_ptr = right.values.as_ptr();

        let left_rows = left.dimensions[0];
        let left_cols = left.dimensions[1];
        let right_cols = right.dimensions[1];

        unsafe {
            let mut row = 0;
            while row != left_rows {
                let mut inner = 0;
                while inner != left_cols {
                    let mut col = 0;
                    while col != right_cols {
                        let left_cell = left_ptr.add(row * left.dimensions[1] + inner);
                        let right_cell = right_ptr.add(inner * right.dimensions[1] + col);
                        let result_cell = result_ptr.add(row * right.dimensions[1] + col);
                        *result_cell += *left_cell * *right_cell;
                        col += 1;
                    }
                    inner += 1;
                }
                row += 1;
            }
        }

        let result = Tensor::new(vec![left.dimensions[0], right.dimensions[1]], result_values);
        Ok(result)
    }
}

impl Into<Vec<f32>> for Tensor {
    fn into(self) -> Vec<f32> {
        self.values
    }
}

impl Display for Tensor {
    // TODO generalize
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        _ = write!(f, "Shape: {:?}", self.dimensions);
        _ = write!(f, "\n");
        for row in 0..self.dimensions[0] {
            for col in 0..self.dimensions[1] {
                let value = self.get(&vec![row, col]);
                if value < 0.0 {
                    _ = write!(f, " {:2.8}", value);
                } else {
                    _ = write!(f, " +{:2.8}", value);
                }
            }
            _ = write!(f, "\n");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::tensor::{Error, Tensor};

    #[test]
    fn new() {
        // Given rows and cols
        // When a matrix is built
        // Then it has the appropriate shape

        let matrix = Tensor::new(
            vec![4, 3],
            vec![
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
            ],
        );
        assert_eq!(matrix.dimensions(), vec![4, 3]);
    }

    #[test]
    fn multiplication_shape_compatibility() {
        // Given two matrices with incompatible shapes
        // When a matrix multiplication is done
        // Then there is an error

        let lhs = Tensor::new(
            vec![1, 1],
            vec![
                0.0, //
            ],
        );

        let rhs = Tensor::new(
            vec![2, 1],
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

        let lhs = Tensor::new(
            vec![3, 2],
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        );
        let rhs = Tensor::new(
            vec![2, 3],
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        );
        let actual_result = &lhs * &rhs;
        let expected_result = Tensor::new(
            vec![3, 3],
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

        let lhs = Tensor::new(
            vec![3, 2],
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        );
        let rhs: Tensor = Tensor::new(
            vec![3, 2],
            vec![
                11.0, 12.0, //
                14.0, 15.0, //
                13.0, 16.0, //
            ],
        );
        let actual_result = &lhs + &rhs;
        let expected_result = Tensor::new(
            vec![3, 2],
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
        let m = Tensor::new(vec![rows, cols], values);
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
        let m = Tensor::new(vec![rows, cols], values);
        let _result = &m + &m;
    }

    #[test]
    fn transpose() {
        let matrix = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix2 = matrix.transpose();
        for row in 0..matrix.dimensions()[0] {
            for col in 0..matrix.dimensions()[1] {
                assert_eq!(matrix2.get(&vec![col, row]), matrix.get(&vec![row, col]));
            }
        }
    }
}
