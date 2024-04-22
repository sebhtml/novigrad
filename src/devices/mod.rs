mod cpu;
pub use cpu::*;
mod cuda;
pub use cuda::*;

use crate::Error;

pub enum Transpose {
    None,
    Ordinary,
    Conjugate,
}

pub trait DeviceInterface {
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
    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
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
    );

    /// SAXPY constant times a vector plus a vector.
    /// y = alpha * x + y
    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32);

    /// SDOT forms the dot product of two vectors.
    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32;

    /// SCOPY copies a vector, x, to a vector, y.
    fn scopy(&self, n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32);

    /// SSCAL scales a vector by a constant.
    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32);
}

pub enum Device {
    Cpu(CpuDevice),
    Cuda(CudaDevice),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu(CpuDevice::default())
    }
    pub fn cuda() -> Result<Self, Error> {
        match CudaDevice::try_default() {
            Ok(cublas) => Ok(Device::Cuda(cublas)),
            Err(error) => Err(error),
        }
    }
}

// TODO add an argument &self to allow to choose between CBlas and CuBlas and the AMD one too.
impl DeviceInterface for Device {
    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
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
        match self {
            Device::Cpu(accelerator) => {
                accelerator.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
            Device::Cuda(accelerator) => {
                accelerator.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
        }
    }

    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
        match self {
            Device::Cpu(accelerator) => accelerator.sdot(n, x, incx, y, incy),
            Device::Cuda(accelerator) => accelerator.sdot(n, x, incx, y, incy),
        }
    }

    fn scopy(&self, n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        match self {
            Device::Cpu(accelerator) => accelerator.scopy(n, x, incx, y, incy),
            Device::Cuda(accelerator) => accelerator.scopy(n, x, incx, y, incy),
        }
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        match self {
            Device::Cpu(accelerator) => accelerator.saxpy(n, alpha, x, incx, y, incy),
            Device::Cuda(accelerator) => accelerator.saxpy(n, alpha, x, incx, y, incy),
        }
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32) {
        match self {
            Device::Cpu(accelerator) => accelerator.sscal(n, alpha, x, incx),
            Device::Cuda(accelerator) => accelerator.sscal(n, alpha, x, incx),
        }
    }
}
