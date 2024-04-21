use cblas::{saxpy, scopy, sdot, sgemm, sscal, Layout, Transpose};

use super::AcceleratorInterface;
extern crate blas_src;

mod tests;

pub struct CBlas {}

impl Default for CBlas {
    fn default() -> Self {
        Self {}
    }
}

impl Into<Transpose> for super::Transpose {
    fn into(self) -> Transpose {
        match self {
            super::Transpose::None => Transpose::None,
            super::Transpose::Ordinary => Transpose::Ordinary,
            super::Transpose::Conjugate => Transpose::Conjugate,
        }
    }
}

impl AcceleratorInterface for CBlas {
    fn sgemm(
        &self,
        transa: super::Transpose,
        transb: super::Transpose,
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
        let transa = transa.into();
        let transb = transb.into();
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
