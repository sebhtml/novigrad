use std::{
    fmt::Display,
    ops::{Add, Mul},
};

#[cfg(test)]
mod tests;

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
        if indices.len() == 3 {
            indices[0] * self.dimensions[1] * self.dimensions[2]
                + indices[1] * self.dimensions[2]
                + indices[2]
        } else if indices.len() == 2 {
            indices[0] * self.dimensions[1] + indices[1]
        } else if indices.len() == 1 {
            indices[0]
        } else {
            usize::MAX
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
    IncompatibleTensorShapes,
}

fn add_matrix_tensor_and_matrix_tensor(left: &Tensor, right: &Tensor) -> Result<Tensor, Error> {
    if left.dimensions != right.dimensions {
        return Err(Error::IncompatibleTensorShapes);
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
        return Err(Error::IncompatibleTensorShapes);
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

// Use broadcasting
fn op_matrix_tensor_and_column_matrix_tensor(
    left: &Tensor,
    right: &Tensor,
    op: &impl F32Op,
) -> Result<Tensor, Error> {
    if !(left.dimensions.len() == 2
        && right.dimensions.len() == 2
        && left.dimensions[0] == right.dimensions[0]
        && left.dimensions[1] != 1
        && right.dimensions[1] == 1)
    {
        return Err(Error::IncompatibleTensorShapes);
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
            let right = right.get(&vec![row, 0]);
            let value = op.op(left, right);
            result.set(&vec![row, col], value);
            col += 1;
        }
        row += 1;
    }
    Ok(result)
}

// Use broadcasting
fn op_tensor_and_matrix(left: &Tensor, right: &Tensor, op: &impl F32Op) -> Result<Tensor, Error> {
    if !(left.dimensions.len() == 3
        && right.dimensions.len() == 2
        && left.dimensions[1] == right.dimensions[0]
        && (left.dimensions[2] == right.dimensions[1] || right.dimensions[1] == 1))
    {
        return Err(Error::IncompatibleTensorShapes);
    }

    let mut values = Vec::new();
    values.resize(left.values.len(), 0.0);

    let mut result = Tensor::new(left.dimensions.clone(), values);

    let subs = left.dimensions[0];
    let rows = left.dimensions[1];
    let cols = left.dimensions[2];
    let mut sub = 0;
    while sub < subs {
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let left_indices = vec![sub, row, col];
                let left = left.get(&left_indices);
                let right_col = if right.dimensions[1] == 1 { 0 } else { col };
                let right = right.get(&vec![row, right_col]);
                let value = op.op(left, right);
                result.set(&left_indices, value);
                col += 1;
            }
            row += 1;
        }
        sub += 1;
    }
    Ok(result)
}

/*
For matrix multiplication:

    (m, n) x (n, r) = (c, m, r)

For 3D tensor multiplication:

    (c, m, n) x (c, n, r) = (c, m, r)

For 4D tensor multiplication:

    (z, c, m, n) x (z, c, n, r) = (z, c, m, r)
*/
// Use broadcasting
fn mul_tensor_and_matrix_dot_product(left: &Tensor, right: &Tensor) -> Result<Tensor, Error> {
    if !(left.dimensions.len() == 3
        && right.dimensions.len() == 2
        && left.dimensions[2] == right.dimensions[0])
    {
        return Err(Error::IncompatibleTensorShapes);
    }

    let result_dimensions = vec![left.dimensions[0], left.dimensions[1], right.dimensions[1]];
    let result_len = result_dimensions[0] * result_dimensions[1] * result_dimensions[2];
    let mut values = Vec::new();
    values.resize(result_len, 0.0);

    let mut result = Tensor::new(result_dimensions.clone(), values);

    let result_subs = result_dimensions[0];
    let result_rows = result_dimensions[1];
    let result_cols = result_dimensions[2];
    let mut result_sub = 0;
    while result_sub < result_subs {
        let mut result_row = 0;
        while result_row < result_rows {
            let mut result_col = 0;
            while result_col < result_cols {
                let mut dot_product = 0.0;
                let left_cols = left.dimensions[2];
                let mut dot_product_iterator = 0;
                while dot_product_iterator < left_cols {
                    let left = left.get(&vec![result_sub, result_row, dot_product_iterator]);
                    let right = right.get(&vec![dot_product_iterator, result_col]);
                    let value = left * right;
                    dot_product += value;
                    dot_product_iterator += 1;
                }
                result.set(&vec![result_sub, result_row, result_col], dot_product);
                result_col += 1;
            }
            result_row += 1;
        }
        result_sub += 1;
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
            Err(Error::IncompatibleTensorShapes)
        }
    }
}

fn multiply_matrix_tensor_and_matrix_tensor(
    left: &Tensor,
    right: &Tensor,
) -> Result<Tensor, Error> {
    if left.dimensions[1] != right.dimensions[0] {
        return Err(Error::IncompatibleTensorShapes);
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
        return Err(Error::IncompatibleTensorShapes);
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
        if left.dimensions.len() == 2
            && right.dimensions.len() == 2
            && left.dimensions[0] == right.dimensions[0]
            && left.dimensions[1] != 1
            && right.dimensions[1] == 1
        {
            op_matrix_tensor_and_column_matrix_tensor(left, right, &F32Mul::default())
        } else if left.dimensions.len() == 3
            && right.dimensions.len() == 2
            && left.dimensions[1] == right.dimensions[0]
            && right.dimensions[1] == 1
        {
            op_tensor_and_matrix(left, right, &F32Mul::default())
        } else if left.dimensions.len() == 3
            && right.dimensions.len() == 2
            && left.dimensions[1] == right.dimensions[0]
            && left.dimensions[2] == right.dimensions[1]
        {
            op_tensor_and_matrix(left, right, &F32Mul::default())
        } else if left.dimensions.len() == 3
            && right.dimensions.len() == 2
            && left.dimensions[2] == right.dimensions[0]
        {
            mul_tensor_and_matrix_dot_product(left, right)
        } else if left.dimensions.len() == 1 && right.dimensions.len() == 1 {
            multiply_vector_tensor_and_vector_tensor(left, right)
        } else if left.dimensions.len() == 2 && right.dimensions.len() == 1 {
            op_matrix_tensor_and_vector_tensor(left, right, &F32Mul::default())
        } else if left.dimensions.len() == 2
            && right.dimensions.len() == 2
            && left.dimensions[1] == right.dimensions[0]
        {
            multiply_matrix_tensor_and_matrix_tensor(left, right)
        } else {
            Err(Error::IncompatibleTensorShapes)
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
