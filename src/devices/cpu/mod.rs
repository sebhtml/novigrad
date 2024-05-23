use std::f32::consts::E;

use cblas::{Layout, Transpose};
extern crate cblas_sys as ffi;
use crate::{DevBufferEnum, Error, GenericTensor};

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
    fn sgemm(
        &self,
        transa: bool,
        transb: bool,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
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

    fn sdot(
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

    fn scopy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        unsafe { ffi::cblas_scopy(n, x, incx, y, incy) }
        Ok(())
    }

    fn saxpy(
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

    fn scalar_mul(&self, n: i32, alpha: *const f32, x: *mut f32, incx: i32) -> Result<(), Error> {
        unsafe { ffi::cblas_sscal(n, *alpha, x, incx) }
        Ok(())
    }

    fn slice(&self, n: i32) -> Result<crate::DevBufferEnum, Error> {
        let len = n as usize;
        let values = vec![0.0; len];
        let slice = DevBufferEnum::CpuBuffer(values);
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

    fn sum(&self, _input: &GenericTensor, _output: &GenericTensor) -> Result<(), Error> {
        todo!()
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
