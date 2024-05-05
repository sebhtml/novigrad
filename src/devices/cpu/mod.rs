use cblas::{Layout, Transpose};
extern crate cblas_sys as ffi;
use crate::{Error, TensorF32};

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
        a: &TensorF32,
        lda: i32,
        b: &TensorF32,
        ldb: i32,
        beta: f32,
        c: &mut TensorF32,
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

    fn sdot(
        &self,
        n: i32,
        x: &TensorF32,
        incx: i32,
        y: &TensorF32,
        incy: i32,
    ) -> Result<f32, Error> {
        let x = x.as_ptr();
        let y = y.as_ptr();
        let result = unsafe { ffi::cblas_sdot(n, x, incx, y, incy) };
        Ok(result)
    }

    fn scopy(
        &self,
        n: i32,
        x: &TensorF32,
        incx: i32,
        y: &mut TensorF32,
        incy: i32,
    ) -> Result<(), Error> {
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        unsafe { ffi::cblas_scopy(n, x, incx, y, incy) }
        Ok(())
    }

    fn saxpy(
        &self,
        n: i32,
        alpha: f32,
        x: &TensorF32,
        incx: i32,
        y: &mut TensorF32,
        incy: i32,
    ) -> Result<(), Error> {
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        unsafe { ffi::cblas_saxpy(n, alpha, x, incx, y, incy) }
        Ok(())
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut TensorF32, incx: i32) -> Result<(), Error> {
        let x = x.as_mut_ptr();
        unsafe { ffi::cblas_sscal(n, alpha, x, incx) }
        Ok(())
    }
}
