use cblas::{saxpy, scopy, sdot, sgemm, sscal, Layout, Transpose};

use super::DeviceInterface;
extern crate blas_src;

mod tests;

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
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    ) {
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
            sgemm(
                layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }

    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
        unsafe { sdot(n, x, incx, y, incy) }
    }

    fn scopy(&self, n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        unsafe { scopy(n, x, incx, y, incy) }
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        unsafe { saxpy(n, alpha, x, incx, y, incy) }
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32) {
        unsafe { sscal(n, alpha, x, incx) }
    }
}
