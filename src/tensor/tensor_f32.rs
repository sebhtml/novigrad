use crate::{
    devices::{Device, DeviceInterface},
    Error,
};
// use rustacuda::memory::cuda_malloc; //TODO use cuda_malloc
use std::{fmt::Display, ops::Mul};

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl Tensor {
    // TODO add device argument
    pub fn new(rows: usize, cols: usize, values: Vec<f32>, _device: &Device) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        Self { rows, cols, values }
    }

    fn operation<Operation>(
        &self,
        device: &Device,
        right: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error>
    where
        Operation: F32Operation,
    {
        let left = self;
        if left.rows != right.rows || left.cols != right.cols {
            return Err(Error::IncompatibleTensorShapes);
        }

        result.reset(left.rows, left.cols, Default::default());
        debug_assert_eq!(result.shape(), left.shape());
        Tensor::scalar_mul(device, 0.0, result);

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

    pub fn assign(&mut self, device: &Device, from: &Tensor) {
        self.reset(from.rows, from.cols, 0.0);
        Tensor::copy(device, from, self);
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

    pub fn mut_values(&mut self) -> &mut Vec<f32> {
        &mut self.values
    }

    // TODO use device for element_wise_mul
    pub fn element_wise_mul(
        &self,
        device: &Device,
        right: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        self.operation::<F32Mul>(device, right, result)
    }

    pub fn dot_product(device: &Device, x: &Tensor, y: &Tensor) -> Result<f32, Error> {
        if x.shape() != y.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let n = x.values.len() as i32;
        let incx = 1;
        let incy = 1;
        Ok(device.sdot(n, x, incx, y, incy))
    }
    fn copy(device: &Device, x: &Tensor, y: &mut Tensor) {
        let n = x.values.len() as i32;
        let incx = 1;
        let incy = 1;
        device.scopy(n, x, incx, y, incy)
    }

    pub fn matmul(
        device: &Device,
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
            device,
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
        device: &Device,
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
            device.sgemm(
                false, false, n as i32, m as i32, k as i32, alpha, b, n as i32, a, k as i32, beta,
                c, n as i32,
            );
            Ok(())
        } else if transa && !transb && !transpose_result {
            if a.rows != b.rows {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.cols, a.rows);

            device.sgemm(
                false,
                true,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b,
                n as i32,
                a,
                a.cols as i32,
                beta,
                c,
                n as i32,
            );

            Ok(())
        } else if !transa && transb && !transpose_result {
            if a.cols != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.rows, b.rows, a.cols);

            device.sgemm(
                true,
                false,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b,
                b.cols as i32,
                a,
                k as i32,
                beta,
                c,
                n as i32,
            );

            Ok(())
        } else if transa && transb && !transpose_result {
            if a.rows != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.rows, a.rows);

            device.sgemm(
                true,
                true,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b,
                b.cols as i32,
                a,
                a.cols as i32,
                beta,
                c,
                n as i32,
            );

            Ok(())
        } else if transa && transb && transpose_result {
            if a.rows != b.cols {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.rows, a.rows);

            device.sgemm(
                false,
                false,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                a.cols as i32,
                b,
                b.cols as i32,
                beta,
                c,
                m as i32,
            );

            Ok(())
        } else if transa && !transb && transpose_result {
            if a.rows != b.rows {
                return Err(Error::IncompatibleTensorShapes);
            }
            let (m, n, k) = (a.cols, b.cols, a.rows);

            device.sgemm(
                false,
                true,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                a.cols as i32,
                b,
                b.cols as i32,
                beta,
                c,
                m as i32,
            );

            Ok(())
        } else {
            Err(Error::UnsupportedOperation)
        }
    }

    pub fn sub(device: &Device, x: &Tensor, y: &mut Tensor) -> Result<(), Error> {
        let alpha = -1.0;
        Self::saxpy(device, alpha, x, y)
    }

    pub fn add(device: &Device, x: &Tensor, y: &mut Tensor) -> Result<(), Error> {
        let alpha = 1.0;
        Self::saxpy(device, alpha, x, y)
    }

    pub fn saxpy(device: &Device, alpha: f32, x: &Tensor, y: &mut Tensor) -> Result<(), Error> {
        if x.values.len() != y.values.len() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let n = x.values.len() as i32;
        let incx = 1;
        let incy = 1;
        device.saxpy(n, alpha, x, incx, y, incy);
        Ok(())
    }

    // TODO use device to clip
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

    pub fn scalar_mul(device: &Device, alpha: f32, x: &mut Tensor) {
        let n = x.values.len() as i32;
        let incx = 1;
        device.sscal(n, alpha, x, incx)
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

impl TryInto<f32> for &Tensor {
    type Error = Error;

    fn try_into(self) -> Result<f32, Self::Error> {
        match self.shape() {
            (1, 1) => Ok(self.get(0, 0)),
            _ => Err(Error::UnsupportedOperation),
        }
    }
}
