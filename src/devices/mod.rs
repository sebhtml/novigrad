mod cpu;
#[cfg(feature = "cuda")]
use crate::Error;
#[cfg(feature = "cuda")]
use rustacuda::memory::CopyDestination;
#[cfg(feature = "cuda")]
use rustacuda::prelude::DeviceBuffer;
use std::borrow::BorrowMut;

pub use cpu::*;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

use crate::Tensor;

#[derive(Debug)]
pub enum DevBuffer {
    CpuBuffer(Vec<f32>),
    #[cfg(feature = "cuda")]
    CudaBuffer(DeviceBuffer<f32>),
}

impl DevBuffer {
    pub fn as_ptr(&self) -> *const f32 {
        match &self {
            DevBuffer::CpuBuffer(ref values) => values.as_ptr(),
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref values) => values.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        match self.borrow_mut() {
            DevBuffer::CpuBuffer(ref mut values) => values.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref mut values) => values.as_mut_ptr(),
        }
    }

    // TODO Delete uses of get_values
    pub fn get_values(&self) -> Vec<f32> {
        match &self {
            DevBuffer::CpuBuffer(ref values) => values.clone(),
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref buffer) => {
                let mut values = vec![0.0; buffer.len()];
                // TODO don't unwrap directly.
                buffer.copy_to(values.as_mut_slice()).unwrap();
                values
            }
        }
    }

    pub fn set_values(&mut self, new_values: Vec<f32>) {
        match self.borrow_mut() {
            DevBuffer::CpuBuffer(ref mut values) => {
                values.clear();
                values.extend_from_slice(new_values.as_slice())
            }
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref mut buffer) => {
                // TODO don't unwrap directly.
                buffer.copy_from(new_values.as_slice()).unwrap();
            }
        }
    }
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
    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
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
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => {
                device.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
        }
    }

    fn sdot(&self, n: i32, x: &Tensor, incx: i32, y: &Tensor, incy: i32) -> f32 {
        match self {
            Device::Cpu(device) => device.sdot(n, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => device.sdot(n, x, incx, y, incy),
        }
    }

    fn scopy(&self, n: i32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32) {
        match self {
            Device::Cpu(device) => device.scopy(n, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => device.scopy(n, x, incx, y, incy),
        }
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &Tensor, incx: i32, y: &mut Tensor, incy: i32) {
        match self {
            Device::Cpu(device) => device.saxpy(n, alpha, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => device.saxpy(n, alpha, x, incx, y, incy),
        }
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut Tensor, incx: i32) {
        match self {
            Device::Cpu(device) => device.sscal(n, alpha, x, incx),
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => device.sscal(n, alpha, x, incx),
        }
    }
}
