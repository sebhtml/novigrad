use crate::dot_product;
use crate::Error;
use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            rows: Default::default(),
            cols: Default::default(),
            values: Default::default(),
        }
    }
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Self {
        Self { rows, cols, values }
    }

    fn operation<Operation>(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error>
    where
        Operation: F32Operation,
    {
        let left = self;
        if left.rows != right.rows || left.cols != right.cols {
            return Err(Error::IncompatibleTensorShapes);
        }

        result.reset(left.rows, left.cols, Default::default());

        let result_ptr = result.values.as_mut_ptr();
        let left_ptr = left.values.as_ptr();
        let right_ptr = right.values.as_ptr();

        unsafe {
            let mut index = 0;
            let len = left.values.len();
            while index < len {
                let left_cell = left_ptr.add(index);
                let right_cell = right_ptr.add(index);
                let result_cell = result_ptr.add(index);
                let left = *left_cell;
                let right = *right_cell;
                let value = Operation::op(left, right);
                debug_assert!(value.is_finite());
                *result_cell = value;
                index += 1;
            }
        }

        Ok(())
    }

    /// lhs not transposed, rhs not transposed, result not transposed.
    fn matmul_lhs_rhs_result(lhs: &Tensor, rhs: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        if lhs.cols != rhs.rows {
            return Err(Error::IncompatibleTensorShapes);
        }

        result.reset(lhs.rows, rhs.cols, Default::default());

        let result_ptr = result.values.as_mut_ptr();
        let left_ptr = lhs.values.as_ptr();
        let right_ptr = rhs.values.as_ptr();

        let left_rows = lhs.rows;
        let left_cols = lhs.cols;
        let right_cols = rhs.cols;

        unsafe {
            let mut row = 0;
            while row != left_rows {
                let mut inner = 0;
                while inner != left_cols {
                    let mut col = 0;
                    while col != right_cols {
                        let left_cell = left_ptr.add(row * left_cols + inner);
                        let right_cell = right_ptr.add(inner * right_cols + col);
                        let result_cell = result_ptr.add(row * right_cols + col);
                        *result_cell += *left_cell * *right_cell;
                        col += 1;
                    }
                    inner += 1;
                }
                row += 1;
            }
        }

        Ok(())
    }

    /// lhs transposed, rhs not transposed, result not transposed.
    fn matmul_lhs_t_rhs_result(
        lhs: &Tensor,
        rhs: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        let lhs_rows = lhs.cols;
        let lhs_cols = lhs.rows;
        if lhs_cols != rhs.rows {
            return Err(Error::IncompatibleTensorShapes);
        }

        result.reset(lhs_rows, rhs.cols, Default::default());

        let result_ptr = result.values.as_mut_ptr();
        let lhs_ptr = lhs.values.as_ptr();
        let right_ptr = rhs.values.as_ptr();

        let right_cols = rhs.cols;

        unsafe {
            let mut row = 0;
            while row != lhs_rows {
                let mut inner = 0;
                while inner != lhs_cols {
                    let mut col = 0;
                    while col != right_cols {
                        //let lhs_value = lhs.get(inner, row);
                        let lhs_value = *lhs_ptr.add(inner * lhs_rows + row);
                        let right_cell = right_ptr.add(inner * right_cols + col);
                        let result_cell = result_ptr.add(row * right_cols + col);
                        *result_cell += lhs_value * *right_cell;
                        col += 1;
                    }
                    inner += 1;
                }
                row += 1;
            }
        }

        Ok(())
    }

    /// lhs transposed, rhs not transposed, result transposed.
    fn matmul_lhs_t_rhs_result_t(
        lhs: &Tensor,
        rhs: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        let lhs_rows = lhs.rows;
        let lhs_cols = lhs.cols;
        if lhs_rows != rhs.rows {
            return Err(Error::IncompatibleTensorShapes);
        }

        result.reset(rhs.cols, lhs_cols, Default::default());

        let lhs_ptr = lhs.values.as_ptr();
        let right_ptr = rhs.values.as_ptr();
        let result_ptr = result.values.as_mut_ptr();

        let right_cols = rhs.cols;

        unsafe {
            let mut lhs_col = 0;
            while lhs_col != lhs_cols {
                let mut inner = 0;
                while inner != lhs_rows {
                    let mut rhs_col = 0;
                    while rhs_col != right_cols {
                        let lhs_value = *lhs_ptr.add(inner * lhs_cols + lhs_col);
                        let right_cell = right_ptr.add(inner * right_cols + rhs_col);
                        let result_cell = result_ptr.add(rhs_col * lhs_cols + lhs_col);
                        *result_cell += lhs_value * *right_cell;
                        rhs_col += 1;
                    }
                    inner += 1;
                }
                lhs_col += 1;
            }
        }

        Ok(())
    }

    /// lhs transposed, rhs transposed, result not transposed.
    fn matmul_lhs_t_rhs_t_result(
        lhs: &Tensor,
        rhs: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        let lhs_rows = lhs.rows;
        let lhs_cols = lhs.cols;
        let rhs_rows = rhs.rows;
        let rhs_cols = rhs.cols;

        if lhs_rows != rhs_cols {
            return Err(Error::IncompatibleTensorShapes);
        }

        result.reset(lhs_cols, rhs_rows, Default::default());

        let mut lhs_col = 0;
        while lhs_col != lhs_cols {
            let mut rhs_row = 0;
            while rhs_row != rhs_rows {
                let mut inner = 0;
                while inner != lhs_rows {
                    let lhs_value = lhs.get(inner, lhs_col);
                    let rhs_value = rhs.get(rhs_row, inner);
                    let old = result.get(lhs_col, rhs_row);
                    let result_value = old + lhs_value * rhs_value;
                    result.set(lhs_col, rhs_row, result_value);
                    inner += 1;
                }
                rhs_row += 1;
            }
            lhs_col += 1;
        }

        Ok(())
    }

    /// lhs transposed, rhs transposed, result transposed.
    fn matmul_lhs_t_rhs_t_result_t(
        lhs: &Tensor,
        rhs: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        let lhs_rows = lhs.rows;
        let lhs_cols = lhs.cols;
        let rhs_rows = rhs.rows;
        let rhs_cols = rhs.cols;

        if lhs_rows != rhs_cols {
            return Err(Error::IncompatibleTensorShapes);
        }

        result.reset(rhs_rows, lhs_cols, Default::default());

        let mut rhs_row = 0;
        while rhs_row != rhs_rows {
            let mut inner = 0;
            while inner != lhs_rows {
                let mut lhs_col = 0;
                while lhs_col != lhs_cols {
                    let lhs_value = lhs.get(inner, lhs_col);
                    let rhs_value = rhs.get(rhs_row, inner);
                    let old = result.get(rhs_row, lhs_col);
                    let result_value = old + lhs_value * rhs_value;
                    result.set(rhs_row, lhs_col, result_value);
                    lhs_col += 1;
                }
                inner += 1;
            }
            rhs_row += 1;
        }

        Ok(())
    }

    /// lhs transposed, rhs transposed, result not transposed.
    /// lhs, rhs, and result are all cache-friendly for the operation.
    fn matmul_lhs_rhs_t_result(
        lhs: &Tensor,
        rhs: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        let lhs_rows = lhs.rows;
        let lhs_cols = lhs.cols;
        let rhs_rows = rhs.rows;
        let rhs_cols = rhs.cols;

        if lhs_cols != rhs_cols {
            return Err(Error::IncompatibleTensorShapes);
        }

        let lhs_slice = lhs.values.as_slice();
        let rhs_slice = rhs.values.as_slice();

        result.reset(lhs_rows, rhs_rows, Default::default());
        let result_slice = result.values.as_mut_slice();

        let mut lhs_row = 0;
        while lhs_row < lhs_rows {
            let lhs_row_index = lhs_row * lhs_cols;
            let lhs_slice = &lhs_slice[lhs_row_index..(lhs_row_index + lhs_cols)];
            let mut rhs_row = 0;
            while rhs_row < rhs_rows {
                let rhs_row_index = rhs_row * rhs_cols;
                let rhs_slice = &rhs_slice[rhs_row_index..(rhs_row_index + lhs_cols)];
                let result_value = &mut result_slice[lhs_row * rhs_rows + rhs_row];
                *result_value = dot_product(lhs_slice, rhs_slice);
                rhs_row += 1;
            }
            lhs_row += 1;
        }

        Ok(())
    }

    fn scalar_op<Operation>(&self, right: f32, result: &mut Tensor) -> Result<(), Error>
    where
        Operation: F32Operation,
    {
        result.reset(self.rows, self.cols, Default::default());
        for i in 0..self.values.len() {
            let left = self.values[i];
            let value = Operation::op(left, right);
            result.values[i] = value;
        }
        Ok(())
    }

    pub fn is_finite(&self) -> bool {
        !self.values.iter().any(|value| !value.is_finite())
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn values<'a>(&'a self) -> &'a Vec<f32> {
        &self.values
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn reset(&mut self, new_rows: usize, new_cols: usize, value: f32) {
        self.rows = new_rows;
        self.cols = new_cols;
        let values = self.rows * self.cols;
        self.values.clear();
        self.values.resize(values, value)
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        let index = self.index(row, col);
        self.values[index]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        let index = self.index(row, col);
        self.values[index] = value;
    }

    pub fn assign(&mut self, from: &Tensor) {
        self.reset(from.rows, from.cols, Default::default());

        let len = from.values.len();
        let mut index = 0;
        while index < len {
            self.values[index] = from.values[index];
            index += 1;
        }
    }

    pub fn row(&self, row: usize, result: &mut Tensor) {
        result.reset(1, self.cols, Default::default());
        for col in 0..self.cols {
            let value = self.get(row, col);
            result.set(0, col, value);
        }
    }

    pub fn col(&self, col: usize, result: &mut Tensor) {
        result.reset(self.rows, 1, Default::default());
        for row in 0..self.rows {
            let value = self.get(row, col);
            result.set(row, 0, value);
        }
    }

    pub fn transpose(&self, other: &mut Tensor) {
        other.reset(self.cols, self.rows, Default::default());
        let rows = self.rows;
        let cols = self.cols;
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let value = self.get(row, col);
                other.set(col, row, value);
                col += 1;
            }
            row += 1;
        }
    }

    pub fn add(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        self.operation::<F32Add>(right, result)
    }

    pub fn sub(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        self.operation::<F32Sub>(right, result)
    }

    pub fn element_wise_mul(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        self.operation::<F32Mul>(right, result)
    }

    pub fn div(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        self.operation::<F32Div>(right, result)
    }

    pub fn matmul(lhs: &Tensor, rhs: &Tensor, result: &mut Tensor, options: u32) -> Result<(), Error> {
        let tranpose_lhs = (options & TRANSPOSE_LHS) > 0;
        let transpose_rhs = (options & TRANSPOSE_RHS) > 0;
        let transpose_result = (options & TRANSPOSE_RESULT) > 0;
        if !tranpose_lhs && !transpose_rhs && !transpose_result {
            Self::matmul_lhs_rhs_result(lhs, rhs, result)
        } else if tranpose_lhs && !transpose_rhs && !transpose_result {
            Self::matmul_lhs_t_rhs_result(lhs, rhs, result)
        } else if !tranpose_lhs && transpose_rhs && !transpose_result {
            Self::matmul_lhs_rhs_t_result(lhs, rhs, result)
        } else if tranpose_lhs && transpose_rhs && !transpose_result {
            Self::matmul_lhs_t_rhs_t_result(lhs, rhs, result)
        } else if tranpose_lhs && transpose_rhs && transpose_result {
            Self::matmul_lhs_t_rhs_t_result_t(lhs, rhs, result)
        } else if tranpose_lhs && !transpose_rhs && transpose_result {
            Self::matmul_lhs_t_rhs_result_t(lhs, rhs, result)
        } else {
            Err(Error::UnsupportedOperation)
        }
    }

    pub fn clip(&self, min: f32, max: f32, result: &mut Tensor) {
        result.reset(self.rows, self.cols, Default::default());
        let len = self.values.len();
        let mut index = 0;
        while index < len {
            let mut value = self.values[index];
            value = value.max(min);
            value = value.min(max);
            result.values[index] = value;
            index += 1;
        }
    }

    pub fn scalar_add(&self, right: f32, result: &mut Tensor) -> Result<(), Error> {
        self.scalar_op::<F32Add>(right, result)
    }

    pub fn scalar_mul(&self, right: f32, result: &mut Tensor) -> Result<(), Error> {
        self.scalar_op::<F32Mul>(right, result)
    }

    pub fn add_to_row(&mut self, row: usize, rhs: &Tensor) -> Result<(), Error> {
        if rhs.cols != self.cols {
            return Err(Error::IncompatibleTensorShapes);
        }

        let mut col = 0;
        let cols = self.cols;
        while col < cols {
            let lhs = self.get(row, col);
            let rhs = rhs.get(0, col);
            self.set(row, col, lhs + rhs);
            col += 1;
        }
        Ok(())
    }

    pub fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error> {
        if (new_rows * new_cols) != self.values.len() {
            return Err(Error::UnsupportedOperation);
        }

        self.rows = new_rows;
        self.cols = new_cols;

        Ok(())
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        _ = write!(f, "Shape: {:?}", (self.rows, self.cols));
        _ = write!(f, "\n");
        for row in 0..self.rows {
            for col in 0..self.cols {
                let value = self.get(row, col);
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

pub const TRANSPOSE_LHS: u32 = 1 << 0;
pub const TRANSPOSE_RHS: u32 = 1 << 1;
pub const TRANSPOSE_RESULT: u32 = 1 << 2;

pub trait F32Operation {
    fn op(left: f32, right: f32) -> f32;
}

struct F32Add {}

impl F32Operation for F32Add {
    fn op(left: f32, right: f32) -> f32 {
        <f32 as Add>::add(left, right)
    }
}

struct F32Sub {}

impl F32Operation for F32Sub {
    fn op(left: f32, right: f32) -> f32 {
        <f32 as Sub>::sub(left, right)
    }
}

struct F32Mul {}

impl F32Operation for F32Mul {
    fn op(left: f32, right: f32) -> f32 {
        <f32 as Mul>::mul(left, right)
    }
}

struct F32Div {}

impl F32Operation for F32Div {
    fn op(left: f32, right: f32) -> f32 {
        <f32 as Div>::div(left, right)
    }
}
