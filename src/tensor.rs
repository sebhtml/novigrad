use std::{
    fmt::Display,
    ops::{Add, Mul},
};

// For broadcasting, see https://medium.com/@hunter-j-phillips/a-simple-introduction-to-broadcasting-db8e581368b3
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
        if indices.len() == 2 {
            indices[0] * self.dimensions[1] + indices[1]
        } else
        /*indices.len() == 1 */
        {
            indices[0]
        }
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
    UnsupportedTensorShapes,
}

fn add_matrix_tensor_and_matrix_tensor(left: &Tensor, right: &Tensor) -> Result<Tensor, Error> {
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

trait F32Op {
    fn op(&self, left: f32, right: f32) -> f32;
}

struct F32Add {}
impl F32Op for F32Add {
    fn op(&self, left: f32, right: f32) -> f32 {
        left + right
    }
}
impl Default for F32Add {
    fn default() -> Self {
        Self {}
    }
}
struct F32Mul {}
impl F32Op for F32Mul {
    fn op(&self, left: f32, right: f32) -> f32 {
        left * right
    }
}
impl Default for F32Mul {
    fn default() -> Self {
        Self {}
    }
}

// Use broadcasting
fn op_matrix_tensor_and_vector_tensor(
    left: &Tensor,
    right: &Tensor,
    op: &impl F32Op,
) -> Result<Tensor, Error> {
    if !(left.dimensions.len() == 2
        && right.dimensions.len() == 1
        && left.dimensions[1] == right.dimensions[0])
    {
        return Err(Error::IncompatibleMatrixShapes);
    }

    let mut values = Vec::new();
    values.resize(left.values.len(), 0.0);

    let mut result = Tensor::new(left.dimensions.clone(), values);

    let rows = left.dimensions[0];
    let cols = left.dimensions[1];
    let mut row = 0;
    while row < rows {
        let mut col = 0;
        while col < cols {
            let left = left.get(&vec![row, col]);
            let right = right.get(&vec![col]);
            let value = op.op(left, right);
            result.set(&vec![row, col], value);
            col += 1;
        }
        row += 1;
    }
    Ok(result)
}

impl Add for &Tensor {
    type Output = Result<Tensor, Error>;

    fn add(self, right: Self) -> Self::Output {
        let left = self;
        if left.dimensions.len() == 2 && right.dimensions.len() == 2 {
            add_matrix_tensor_and_matrix_tensor(left, right)
        } else if left.dimensions.len() == 2 && right.dimensions.len() == 1 {
            op_matrix_tensor_and_vector_tensor(left, right, &F32Add::default())
        } else {
            Err(Error::UnsupportedTensorShapes)
        }
    }
}

fn multiply_matrix_tensor_and_matrix_tensor(
    left: &Tensor,
    right: &Tensor,
) -> Result<Tensor, Error> {
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

fn multiply_vector_tensor_and_vector_tensor(
    left: &Tensor,
    right: &Tensor,
) -> Result<Tensor, Error> {
    if !(left.dimensions.len() == 1
        && right.dimensions.len() == 1
        && left.dimensions[0] != 1
        && right.dimensions[0] == 1)
    {
        return Err(Error::IncompatibleMatrixShapes);
    }

    let mut values = Vec::new();
    values.resize(left.values.len(), 0.0);

    let mut result = Tensor::new(left.dimensions.clone(), values);

    let rows = left.dimensions[0];
    let mut row = 0;
    while row < rows {
        let value = left.get(&vec![row]) * right.get(&vec![0]);
        result.set(&vec![row], value);
        row += 1;
    }
    Ok(result)
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
        if left.dimensions.len() == 2 && right.dimensions.len() == 2 {
            multiply_matrix_tensor_and_matrix_tensor(left, right)
        } else if left.dimensions.len() == 1 && right.dimensions.len() == 1 {
            multiply_vector_tensor_and_vector_tensor(left, right)
        } else if left.dimensions.len() == 2 && right.dimensions.len() == 1 {
            op_matrix_tensor_and_vector_tensor(left, right, &F32Mul::default())
        } else {
            Err(Error::UnsupportedTensorShapes)
        }
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

    #[test]
    fn add_matrix_tensor_and_vector_tensor() {
        let tensor1 = Tensor::new(
            vec![4, 3],
            vec![
                //
                0.0, 0.0, 0.0, //
                10.0, 10.0, 10.0, //
                20.0, 20.0, 20.0, //
                30.0, 30.0, 30.0, //
            ],
        );
        let tensor2 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let result = &tensor1 + &tensor2;
        let expected = Tensor::new(
            vec![4, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                11.0, 12.0, 13.0, //
                21.0, 22.0, 23.0, //
                31.0, 32.0, 33.0, //
            ],
        );
        assert_eq!(result, Ok(expected));
    }

    #[test]
    fn multiply_vector_tensor_and_vector_tensor() {
        let tensor1 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let tensor2 = Tensor::new(vec![1], vec![2.0]);
        assert_eq!(
            &tensor1 * &tensor2,
            Ok(Tensor::new(vec![3], vec![2.0, 4.0, 6.0,],))
        );
    }

    #[test]
    fn multiply_matrix_tensor_and_vector_tensor() {
        let tensor1 = Tensor::new(
            vec![3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        );
        let tensor2 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let expected = Tensor::new(
            vec![3, 3],
            vec![
                //
                1.0, 4.0, 9.0, //
                4.0, 10.0, 18.0, //
                7.0, 16.0, 27.0, //
            ],
        );
        assert_eq!(&tensor1 * &tensor2, Ok(expected));
    }
}
