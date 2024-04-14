use crate::{
    accelerator::{AcceleratorInterface, Blas, Layout, Transpose},
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

    pub fn assign(&mut self, blas: &Blas, from: &Tensor) {
        self.reset(from.rows, from.cols, 0.0);
        Tensor::scopy(blas, from, self);
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

    // TODO use blas or something else for element_wise_mul
    pub fn element_wise_mul(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        self.operation::<F32Mul>(right, result)
    }

    pub fn sdot(blas: &Blas, x: &Tensor, y: &Tensor) -> Result<f32, Error> {
        if x.shape() != y.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let n = x.values.len() as i32;
        let x = &x.values;
        let incx = 1;
        let y = &y.values;
        let incy = 1;
        Ok(blas.sdot(n, x, incx, y, incy))
    }
    pub fn scopy(blas: &Blas, x: &Tensor, y: &mut Tensor) {
        let n = x.values.len() as i32;
        let x = &x.values;
        let incx = 1;
        let y = &mut y.values;
        let incy = 1;
        blas.scopy(n, x, incx, y, incy)
    }
    ///  SGEMM  performs one of the matrix-matrix operations
    /// https://netlib.org/lapack/explore-html-3.6.1/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
    ///
    /// C := alpha * op(A) * op(B) + beta * C,
    ///
    /// where  op(X) is one of
    ///    op(X) = X   or   op(X) = X^T,
    ///
    /// alpha and beta are scalars.
    /// A, B and C are matrices.
    ///
    /// op(A) is an m by k matrix
    /// op(B) is a k by n matrix
    /// C is an m by n matrix.
    ///
    pub fn sgemm(
        blas: &Blas,
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
            blas.sgemm(
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

            blas.sgemm(
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

            blas.sgemm(
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

            blas.sgemm(
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

            blas.sgemm(
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

            blas.sgemm(
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

    // SAXPY constant times a vector plus a vector.
    // y = alpha * x + y
    pub fn saxpy(blas: &Blas, alpha: f32, x: &Tensor, y: &mut Tensor) -> Result<(), Error> {
        if x.values.len() != y.values.len() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let n = x.values.len() as i32;
        let x = &x.values;
        let incx = 1;
        let y = &mut y.values;
        let incy = 1;
        blas.saxpy(n, alpha, x, incx, y, incy);
        Ok(())
    }

    // TODO use blas to clip
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

    pub fn sscal(blas: &Blas, alpha: f32, x: &mut Tensor) {
        let n = x.values.len() as i32;
        let x = &mut x.values;
        let incx = 1;
        blas.sscal(n, alpha, x, incx)
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
