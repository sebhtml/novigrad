use crate::{
    accelerator::{Accelerator, AcceleratorInterface, Layout, Transpose},
    Error,
};
use std::{fmt::Display, ops::Mul};

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
        debug_assert_eq!(values.len(), rows * cols);
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

    pub fn assign(&mut self, accelerator: &Accelerator, from: &Tensor) {
        self.reset(from.rows, from.cols, 0.0);
        Tensor::scopy(accelerator, from, self);
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

    pub fn values(&self) -> &Vec<f32> {
        &self.values
    }

    // TODO use accelerator for element_wise_mul
    pub fn element_wise_mul(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        self.operation::<F32Mul>(right, result)
    }

    pub fn dot_product(accelerator: &Accelerator, x: &Tensor, y: &Tensor) -> Result<f32, Error> {
        if x.shape() != y.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let n = x.values.len() as i32;
        let x = &x.values;
        let incx = 1;
        let y = &y.values;
        let incy = 1;
        Ok(accelerator.sdot(n, x, incx, y, incy))
    }
    pub fn scopy(accelerator: &Accelerator, x: &Tensor, y: &mut Tensor) {
        let n = x.values.len() as i32;
        let x = &x.values;
        let incx = 1;
        let y = &mut y.values;
        let incy = 1;
        accelerator.scopy(n, x, incx, y, incy)
    }

    pub fn matmul(
        accelerator: &Accelerator,
        transa: bool,
        transb: bool,
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        transpose_result: bool,
    ) -> Result<(), Error> {
        let alpha = 1.0;
        let beta = 0.0;
        Tensor::gemm(
            accelerator,
            transa,
            transb,
            alpha,
            a,
            b,
            beta,
            c,
            transpose_result,
        )
    }

    pub fn gemm(
        accelerator: &Accelerator,
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
            accelerator.sgemm(
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
            Ok(())
        } else if transa && !transb && !transpose_result {
            if a.rows != b.rows {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.cols, a.rows);

            accelerator.sgemm(
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

            Ok(())
        } else if !transa && transb && !transpose_result {
            if a.cols != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.rows, b.rows, a.cols);

            accelerator.sgemm(
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

            Ok(())
        } else if transa && transb && !transpose_result {
            if a.rows != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.rows, a.rows);

            accelerator.sgemm(
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

            Ok(())
        } else if transa && transb && transpose_result {
            if a.rows != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.rows, a.rows);

            accelerator.sgemm(
                Layout::ColumnMajor,
                Transpose::None,
                Transpose::None,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                &a.values,
                a.cols as i32,
                &b.values,
                b.cols as i32,
                beta,
                &mut c.values,
                m as i32,
            );

            Ok(())
        } else if transa && !transb && transpose_result {
            if a.rows != b.rows {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.cols, a.rows);

            accelerator.sgemm(
                Layout::ColumnMajor,
                Transpose::None,
                Transpose::Ordinary,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                &a.values,
                a.cols as i32,
                &b.values,
                b.cols as i32,
                beta,
                &mut c.values,
                m as i32,
            );

            Ok(())
        } else {
            Err(Error::UnsupportedOperation)
        }
    }

    pub fn sub(accelerator: &Accelerator, x: &Tensor, y: &mut Tensor) -> Result<(), Error> {
        let alpha = -1.0;
        Self::saxpy(accelerator, alpha, x, y)
    }

    pub fn add(accelerator: &Accelerator, x: &Tensor, y: &mut Tensor) -> Result<(), Error> {
        let alpha = 1.0;
        Self::saxpy(accelerator, alpha, x, y)
    }

    pub fn saxpy(
        accelerator: &Accelerator,
        alpha: f32,
        x: &Tensor,
        y: &mut Tensor,
    ) -> Result<(), Error> {
        if x.values.len() != y.values.len() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let n = x.values.len() as i32;
        let x = &x.values;
        let incx = 1;
        let y = &mut y.values;
        let incy = 1;
        accelerator.saxpy(n, alpha, x, incx, y, incy);
        Ok(())
    }

    // TODO use accelerator to clip
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

    pub fn scalar_mul(accelerator: &Accelerator, alpha: f32, x: &mut Tensor) {
        let n = x.values.len() as i32;
        let x = &mut x.values;
        let incx = 1;
        accelerator.sscal(n, alpha, x, incx)
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

struct F32Mul {}

impl F32Operation for F32Mul {
    fn op(left: f32, right: f32) -> f32 {
        <f32 as Mul>::mul(left, right)
    }
}
