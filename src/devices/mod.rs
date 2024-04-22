mod cpu;
pub use cpu::*;
mod cuda;
pub use cuda::*;

use crate::{Error, Tensor};

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
    );

    /// SAXPY constant times a vector plus a vector.
    /// y = alpha * x + y
    fn saxpy(&self, n: i32, alpha: f32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32);

    /// SDOT forms the dot product of two vectors.
    fn sdot(&self, n: i32, x: &Tensor, incx: i32, y: &Tensor, incy: i32) -> f32;

    /// SCOPY copies a vector, x, to a vector, y.
    fn scopy(&self, n: i32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32);

    /// SSCAL scales a vector by a constant.
    fn sscal(&self, n: i32, alpha: f32, x: &mut Tensor, incx: i32);
}

pub enum Device {
    Cpu(CpuDevice),
    Cuda(CudaDevice),
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
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
    pub fn tensor(&self, rows: usize, cols: usize, values: Vec<f32>) -> Tensor {
        Tensor::new(rows, cols, values, self)
    }
}

// TODO add an argument &self to allow to choose between CBlas and CuBlas and the AMD one too.
impl DeviceInterface for Device {
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
        match self {
            Device::Cpu(device) => {
                device.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
            Device::Cuda(device) => {
                device.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
        }
    }

    fn sdot(&self, n: i32, x: &Tensor, incx: i32, y: &Tensor, incy: i32) -> f32 {
        match self {
            Device::Cpu(device) => device.sdot(n, x, incx, y, incy),
            Device::Cuda(device) => device.sdot(n, x, incx, y, incy),
        }
    }

    fn scopy(&self, n: i32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32) {
        match self {
            Device::Cpu(device) => device.scopy(n, x, incx, y, incy),
            Device::Cuda(device) => device.scopy(n, x, incx, y, incy),
        }
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32) {
        match self {
            Device::Cpu(device) => device.saxpy(n, alpha, x, incx, y, incy),
            Device::Cuda(device) => device.saxpy(n, alpha, x, incx, y, incy),
        }
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut Tensor, incx: i32) {
        match self {
            Device::Cpu(device) => device.sscal(n, alpha, x, incx),
            Device::Cuda(device) => device.sscal(n, alpha, x, incx),
        }
    }
}
