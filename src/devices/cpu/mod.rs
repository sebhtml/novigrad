use cblas::{saxpy, scopy, sdot, sgemm, sscal, Layout, Transpose};

use crate::Tensor;

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
        a: &Tensor,
        lda: i32,
        b: &Tensor,
        ldb: i32,
        beta: f32,
        c: &mut Tensor,
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
        let a = a.values();
        let b = b.values();
        let c = c.mut_values();
        unsafe {
            sgemm(
                layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }

    fn sdot(&self, n: i32, x: &Tensor, incx: i32, y: &Tensor, incy: i32) -> f32 {
        let x = x.values();
        let y = y.values();
        unsafe { sdot(n, x, incx, y, incy) }
    }

    fn scopy(&self, n: i32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32) {
        let x = x.values();
        let y = y.mut_values();
        unsafe { scopy(n, x, incx, y, incy) }
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32) {
        let x = x.values();
        let y = y.mut_values();
        unsafe { saxpy(n, alpha, x, incx, y, incy) }
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut Tensor, incx: i32) {
        let x = x.mut_values();
        unsafe { sscal(n, alpha, x, incx) }
    }
}
