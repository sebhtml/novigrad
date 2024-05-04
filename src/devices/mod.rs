mod cpu;
#[cfg(feature = "cuda")]
use crate::Error;
#[cfg(feature = "cuda")]
use rustacuda::memory::CopyDestination;
#[cfg(feature = "cuda")]
use rustacuda::prelude::DeviceBuffer;
use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    mem::swap,
    rc::Rc,
};

pub use cpu::*;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

use crate::{OperatorTrait, Tensor, TensorF32};

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

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        match &self {
            DevBuffer::CpuBuffer(ref values) => Ok(values.clone()),
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref buffer) => {
                let mut values = vec![0.0; buffer.len()];
                match buffer.copy_to(values.as_mut_slice()) {
                    Ok(_) => Ok(values),
                    _ => Err(Error::UnsupportedOperation),
                }
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

    pub fn len(&self) -> usize {
        match self {
            DevBuffer::CpuBuffer(buffer) => buffer.len(),
            DevBuffer::CudaBuffer(buffer) => buffer.len(),
        }
    }

    pub fn resize(&mut self, new_len: usize) {
        match self {
            DevBuffer::CpuBuffer(buffer) => buffer.resize(new_len, Default::default()),
            DevBuffer::CudaBuffer(buffer) => {
                if buffer.len() != new_len {
                    let mut new_buffer = unsafe { DeviceBuffer::uninitialized(new_len).unwrap() };
                    swap(buffer, &mut new_buffer);
                }
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
        a: &TensorF32,
        lda: i32,
        b: &TensorF32,
        ldb: i32,
        beta: f32,
        c: &mut TensorF32,
        ldc: i32,
    ) -> Result<(), Error>;

    /// SAXPY constant times a vector plus a vector.
    /// y = alpha * x + y
    fn saxpy(
        &self,
        n: i32,
        alpha: f32,
        x: &TensorF32,
        incx: i32,
        y: &mut TensorF32,
        incy: i32,
    ) -> Result<(), Error>;

    /// SDOT forms the dot product of two vectors.
    fn sdot(
        &self,
        n: i32,
        x: &TensorF32,
        incx: i32,
        y: &TensorF32,
        incy: i32,
    ) -> Result<f32, Error>;

    /// SCOPY copies a vector, x, to a vector, y.
    fn scopy(
        &self,
        n: i32,
        x: &TensorF32,
        incx: i32,
        y: &mut TensorF32,
        incy: i32,
    ) -> Result<(), Error>;

    /// SSCAL scales a vector by a constant.
    fn sscal(&self, n: i32, alpha: f32, x: &mut TensorF32, incx: i32) -> Result<(), Error>;
}

pub struct Device {
    tensors_with_requires_grad: RefCell<Vec<Tensor>>,
    device: DeviceEnum,
}

pub enum DeviceEnum {
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
    pub fn new(device: DeviceEnum) -> Self {
        Self {
            tensors_with_requires_grad: vec![].into(),
            device,
        }
    }
    pub fn cpu() -> Self {
        Self::new(DeviceEnum::Cpu(CpuDevice::default()))
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self, Error> {
        match CudaDevice::try_default() {
            Ok(cublas) => Ok(Self::new(DeviceEnum::Cuda(cublas))),
            Err(error) => Err(error),
        }
    }

    pub fn tensor_f32(&self, rows: usize, cols: usize, values: Vec<f32>) -> TensorF32 {
        TensorF32::new(rows, cols, values, self)
    }

    pub fn tensor(
        &self,
        operator: Rc<dyn OperatorTrait>,
        inputs: &[Tensor],
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        requires_grad: bool,
    ) -> Tensor {
        let len = rows * cols;
        let tensor = Tensor::new(
            operator,
            inputs,
            Rc::new(RefCell::new(Self::tensor_f32(&self, rows, cols, values))),
            Rc::new(RefCell::new(Self::tensor_f32(
                &self,
                rows,
                cols,
                vec![0.0; len],
            ))),
        );
        if requires_grad {
            self.tensors_with_requires_grad
                .borrow_mut()
                .push(tensor.clone())
        }
        tensor
    }

    pub fn tensors_with_requires_grad(&self) -> Vec<Tensor> {
        self.tensors_with_requires_grad.borrow().clone()
    }

    pub fn buffer(&self, values: Vec<f32>) -> DevBuffer {
        match self.device {
            DeviceEnum::Cpu(_) => DevBuffer::CpuBuffer(values),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(_) => {
                // TODO don't unwrap
                let mut buffer = unsafe { DeviceBuffer::uninitialized(values.len()).unwrap() };
                buffer.copy_from(values.as_slice()).unwrap();
                DevBuffer::CudaBuffer(buffer)
            }
        }
    }
}

impl DeviceInterface for Device {
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
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => {
                device.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => {
                device.sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
        }
    }

    fn sdot(
        &self,
        n: i32,
        x: &TensorF32,
        incx: i32,
        y: &TensorF32,
        incy: i32,
    ) -> Result<f32, Error> {
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => device.sdot(n, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => device.sdot(n, x, incx, y, incy),
        }
    }

    fn scopy(
        &self,
        n: i32,
        x: &TensorF32,
        incx: i32,
        y: &mut TensorF32,
        incy: i32,
    ) -> Result<(), Error> {
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => device.scopy(n, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => device.scopy(n, x, incx, y, incy),
        }
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
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => device.saxpy(n, alpha, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => device.saxpy(n, alpha, x, incx, y, incy),
        }
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut TensorF32, incx: i32) -> Result<(), Error> {
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => device.sscal(n, alpha, x, incx),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => device.sscal(n, alpha, x, incx),
        }
    }
}
