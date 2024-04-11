use crate::Error;
use cblas::*;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
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

    pub fn gemm(
        transa: bool,
        transb: bool,
        alpha: f32,
        a: &Tensor,
        b: &Tensor,
        beta: f32,
        c: &mut Tensor,
        transpose_result: bool,
    ) -> Result<(), Error> {
        if !transa && !transb && !transpose_result {
            if a.cols != b.rows {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.rows, b.cols, a.cols);
            c.reset(m as usize, n as usize, Default::default());
            unsafe {
                sgemm(
                    Layout::ColumnMajor,
                    Transpose::None,
                    Transpose::None,
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha,
                    &b.values,
                    n as i32,
                    &a.values,
                    k as i32,
                    beta,
                    &mut c.values,
                    n as i32,
                );
            }
            Ok(())
        } else if transa && !transb && !transpose_result {
            if a.rows != b.rows {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.cols, a.rows);
            c.reset(m as usize, n as usize, Default::default());
            unsafe {
                sgemm(
                    Layout::ColumnMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha,
                    &b.values,
                    n as i32,
                    &a.values,
                    a.cols as i32,
                    beta,
                    &mut c.values,
                    n as i32,
                );
            }
            Ok(())
        } else if !transa && transb && !transpose_result {
            if a.cols != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.rows, b.rows, a.cols);
            c.reset(m as usize, n as usize, Default::default());
            unsafe {
                sgemm(
                    Layout::ColumnMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha,
                    &b.values,
                    b.cols as i32,
                    &a.values,
                    k as i32,
                    beta,
                    &mut c.values,
                    n as i32,
                );
            }
            Ok(())
        } else if transa && transb && !transpose_result {
            if a.rows != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.rows, a.rows);
            c.reset(m as usize, n as usize, Default::default());
            unsafe {
                sgemm(
                    Layout::ColumnMajor,
                    Transpose::Ordinary,
                    Transpose::Ordinary,
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha,
                    &b.values,
                    b.cols as i32,
                    &a.values,
                    a.cols as i32,
                    beta,
                    &mut c.values,
                    n as i32,
                );
            }
            Ok(())
        } else if transa && transb && transpose_result {
            let mut tmp = Tensor::default();
            Self::gemm(transa, transb, alpha, a, b, beta, &mut tmp, false)?;
            tmp.transpose(c);
            Ok(())
        } else if transa && !transb && transpose_result {
            let mut tmp = Tensor::default();
            Self::gemm(transa, transb, alpha, a, b, beta, &mut tmp, false)?;
            tmp.transpose(c);
            Ok(())
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

    pub fn scalar_mul(&self, right: f32, result: &mut Tensor) -> Result<(), Error> {
        // TODO use gemm with identity matrix and alpha = -learning_rate
        self.scalar_op::<F32Mul>(right, result)
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
