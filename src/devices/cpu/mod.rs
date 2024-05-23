use std::f32::consts::E;

use cblas::{Layout, Transpose};
extern crate cblas_sys as ffi;
use crate::{error, DevSliceEnum, Error, ErrorEnum, Tensor};

use super::DeviceInterface;
extern crate blas_src;

mod tests;

#[derive(Debug)]
pub struct CpuDevice {}

impl Default for CpuDevice {
    fn default() -> Self {
        Self {}
    }
}

impl DeviceInterface for CpuDevice {
    fn gemm(
        &self,
        transa: bool,
        transb: bool,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Tensor,
        a: &Tensor,
        lda: i32,
        b: &Tensor,
        ldb: i32,
        beta: &Tensor,
        c: &Tensor,
        ldc: i32,
    ) -> Result<(), Error> {
        let layout = Layout::ColumnMajor;
        let transa = match transa {
            false => Transpose::None,
            true => Transpose::Ordinary,
        };
        let transb = match transb {
            false => Transpose::None,
            true => Transpose::Ordinary,
        };
        let alpha = alpha.get_values()?;
        let alpha = alpha[0];
        let beta = beta.get_values()?;
        let beta = beta[0];
        let a = a.as_ptr();
        let b = b.as_ptr();
        let c = c.as_mut_ptr();
        unsafe {
            ffi::cblas_sgemm(
                layout.into(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            )
        }
        Ok(())
    }

    fn dot(
        &self,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Result<f32, Error> {
        let result = unsafe { ffi::cblas_sdot(n, x, incx, y, incy) };
        Ok(result)
    }

    fn copy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        unsafe { ffi::cblas_scopy(n, x, incx, y, incy) }
        Ok(())
    }

    fn axpy(
        &self,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> Result<(), Error> {
        unsafe { ffi::cblas_saxpy(n, alpha, x, incx, y, incy) }
        Ok(())
    }

    fn scalar_mul(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error> {
        let n = x.len() as i32;
        let x = x.as_mut_ptr();
        let incx = 1;
        let alpha = alpha.get_values()?;
        let alpha = alpha[0];
        unsafe { ffi::cblas_sscal(n, alpha, x, incx) }
        Ok(())
    }

    fn slice(&self, n: i32) -> Result<crate::DevSliceEnum, Error> {
        let len = n as usize;
        let values = vec![0.0; len];
        let slice = DevSliceEnum::CpuDevSlice(values);
        Ok(slice)
    }

    fn softmax(
        &self,
        rows: i32,
        cols: i32,
        input: *const f32,
        output: *mut f32,
    ) -> Result<(), Error> {
        CpuDevice::_softmax(rows, cols, input, output)
    }

    fn sum(&self, _input: &Tensor, _output: &Tensor) -> Result<(), Error> {
        todo!()
    }

    fn mul(&self, left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error> {
        if left.size() != right.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }

        debug_assert_eq!(result.size(), left.size());

        let mut result_values = result.get_values()?;
        let left_values = left.get_values()?;
        let right_values = right.get_values()?;

        let result_ptr = result_values.as_mut_ptr();
        let left_ptr = left_values.as_ptr();
        let right_ptr = right_values.as_ptr();

        unsafe {
            let mut index = 0;
            let len = left_values.len();
            while index < len {
                let left_cell = left_ptr.add(index);
                let right_cell = right_ptr.add(index);
                let result_cell = result_ptr.add(index);
                let left = *left_cell;
                let right = *right_cell;
                let value = left * right;
                *result_cell = value;
                index += 1;
            }
        }

        result.set_values(result_values)
    }

    fn sigmoid(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let rows = input.rows();
        let cols = input.cols();
        let values = input.get_values()?;
        let mut result_values = output.get_values()?;
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = values[input.index(row, col)];
                let y = 1.0 / (1.0 + E.powf(-x));
                result_values[output.index(row, col)] = y;
                col += 1;
            }
            row += 1;
        }
        output.set_values(result_values)
    }
}

impl CpuDevice {
    pub fn _softmax(
        rows: i32,
        cols: i32,
        input: *const f32,
        output: *mut f32,
    ) -> Result<(), Error> {
        let rows = rows as usize;
        let cols = cols as usize;
        let mut row = 0;
        while row < rows {
            // Find max

            let mut max = unsafe { *input.add(row * cols + 0) };
            let mut col = 0;
            while col < cols {
                let x = unsafe { *input.add(row * cols + col) };
                max = max.max(x);
                col += 1;
            }

            // For each value:
            // 1. substract the max
            // 2. compute E^x
            // 3. add result to sum
            let mut sum = 0.0;
            let mut col = 0;
            while col < cols {
                let x = unsafe { *input.add(row * cols + col) };
                debug_assert_eq!(false, x.is_nan());
                let y = E.powf(x - max);
                debug_assert_eq!(false, y.is_nan(), "x: {}, max: {}, y: {}", x, max, y,);
                unsafe { *output.add(row * cols + col) = y };
                sum += y;
                col += 1;
            }

            // Divide every value by sum.
            let mut col = 0;
            while col < cols {
                let x = unsafe { *output.add(row * cols + col) };
                debug_assert_eq!(false, x.is_nan());
                debug_assert_ne!(0.0, sum);
                let y = x / sum;
                debug_assert_eq!(false, y.is_nan());
                unsafe { *output.add(row * cols + col) = y };
                col += 1;
            }
            row += 1;
        }

        Ok(())
    }
}
