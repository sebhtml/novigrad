use cblas::{Layout, Transpose};
extern crate cblas_sys as ffi;
use crate::{DevBufferEnum, Error};

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

    fn sscal(&self, n: i32, alpha: f32, x: *mut f32, incx: i32) -> Result<(), Error> {
        unsafe { ffi::cblas_sscal(n, alpha, x, incx) }
        Ok(())
    }

    fn slice(&self, n: i32) -> Result<crate::DevBufferEnum, Error> {
        let len = n as usize;
        let values = vec![0.0; len];
        let slice = DevBufferEnum::CpuBuffer(values);
        Ok(slice)
    }
}
