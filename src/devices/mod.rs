mod cpu;
use crate::Error;
use std::{
    borrow::Borrow,
    cell::RefCell,
    collections::{HashMap, LinkedList},
    mem::swap,
    ops::Deref,
    rc::Rc,
};

pub use cpu::*;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

use crate::{Tensor, TensorF32};
mod buffer;
pub use buffer::*;

pub struct MemoryInfo {
    pub used: usize,
    pub free: usize,
    pub total: usize,
    pub model_parameters: usize,
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
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    ) -> Result<(), Error>;

    /// SAXPY constant times a vector plus a vector.
    /// y = alpha * x + y
    fn saxpy(
        &self,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> Result<(), Error>;

    /// SDOT forms the dot product of two vectors.
    fn sdot(
        &self,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Result<f32, Error>;

    /// SCOPY copies a vector, x, to a vector, y.
    fn scopy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error>;

    /// SSCAL scales a vector by a constant.
    fn sscal(&self, n: i32, alpha: f32, x: *mut f32, incx: i32) -> Result<(), Error>;
}

#[derive(Clone, Debug)]
pub struct Device {
    next_name: Rc<RefCell<usize>>,
    used: Rc<RefCell<usize>>,
    tensors_to_optimize: Rc<RefCell<Vec<Tensor>>>,
    device: Rc<DeviceEnum>,
    available_buffers: Rc<RefCell<HashMap<usize, LinkedList<DevBuffer>>>>,
}

#[derive(Debug)]
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
            next_name: Default::default(),
            used: Default::default(),
            tensors_to_optimize: Rc::new(RefCell::new(vec![])),
            device: Rc::new(device),
            available_buffers: Default::default(),
        }
    }

    pub fn cpu() -> Self {
        Self::new(DeviceEnum::Cpu(CpuDevice::default()))
    }

    pub fn recycle(&self, len: usize, buffer: &mut DevBuffer) {
        let mut recycled_buffer = DevBuffer::new(self, 0);
        swap(&mut recycled_buffer, buffer);

        let available_buffers: &mut HashMap<_, _> =
            &mut self.available_buffers.deref().borrow_mut();
        let entry = available_buffers.entry(len);
        entry.or_default().push_back(recycled_buffer)
    }

    pub fn get_memory_info(&self) -> Result<MemoryInfo, Error> {
        let mut model_parameters = 0;
        let gradients = self.tensors_to_optimize();
        for gradient in gradients.deref().borrow().iter() {
            let tensor_len = gradient.tensor().deref().borrow().len();
            model_parameters += tensor_len;
        }
        Ok(MemoryInfo {
            used: *self.used.deref().borrow(),
            free: 0,
            total: 0,
            model_parameters,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self, Error> {
        match CudaDevice::try_default() {
            Ok(cublas) => Ok(Self::new(DeviceEnum::Cuda(cublas))),
            Err(error) => Err(error),
        }
    }

    pub fn tensor_f32(&self, rows: usize, cols: usize, values: Vec<f32>) -> TensorF32 {
        let name = *self.next_name.deref().borrow();
        *self.next_name.deref().borrow_mut() += 1;
        TensorF32::new(name, rows, cols, values, self)
    }

    pub fn tensor(
        &self,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        inputs: &[&Tensor],
        requires_grad: bool,
        optimize: bool,
    ) -> Tensor {
        let name = *self.next_name.deref().borrow();
        *self.next_name.deref().borrow_mut() += 1;
        let len = rows * cols;
        let gradient = if requires_grad {
            Self::tensor_f32(&self, rows, cols, vec![0.0; len])
        } else {
            Self::tensor_f32(&self, 0, 0, vec![])
        };
        let tensor = Self::tensor_f32(&self, rows, cols, values);
        let tensor = Tensor::new(name, tensor, gradient, inputs);
        if optimize {
            self.tensors_to_optimize
                .deref()
                .borrow_mut()
                .push(tensor.clone())
        }
        tensor
    }

    pub fn tensor_count(&self) -> usize {
        *self.next_name.deref().borrow()
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        for t in self.tensors_to_optimize.deref().borrow().iter() {
            count += t.tensor().deref().borrow().len();
        }
        count
    }

    pub fn tensors_to_optimize(&self) -> &Rc<RefCell<Vec<Tensor>>> {
        &self.tensors_to_optimize
    }

    pub fn zero_grad(&self) -> Result<(), Error> {
        let gradients: &[Tensor] = &self.tensors_to_optimize().deref().borrow();
        for gradient in gradients {
            let gradient: &mut TensorF32 = &mut gradient.gradient().deref().borrow_mut();
            gradient.zero()?;
        }
        Ok(())
    }

    pub fn buffer(&self, len: usize) -> DevBuffer {
        let recycled = self
            .available_buffers
            .deref()
            .borrow_mut()
            .get_mut(&len)
            .map(|x| x.pop_back())
            .flatten();
        match recycled {
            Some(buffer) => {
                //println!("Recycled buffer with length {}", len);
                buffer
            }
            None => {
                let used: &mut usize = &mut self.used.deref().borrow_mut();
                *used += len;
                DevBuffer::new(self, len)
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
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
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
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Result<f32, Error> {
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => device.sdot(n, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => device.sdot(n, x, incx, y, incy),
        }
    }

    fn scopy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
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
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> Result<(), Error> {
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => device.saxpy(n, alpha, x, incx, y, incy),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => device.saxpy(n, alpha, x, incx, y, incy),
        }
    }

    fn sscal(&self, n: i32, alpha: f32, x: *mut f32, incx: i32) -> Result<(), Error> {
        match self.device.borrow() {
            DeviceEnum::Cpu(device) => device.sscal(n, alpha, x, incx),
            #[cfg(feature = "cuda")]
            DeviceEnum::Cuda(device) => device.sscal(n, alpha, x, incx),
        }
    }
}
