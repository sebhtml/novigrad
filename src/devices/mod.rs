mod cpu;
use crate::Error;
use std::{
    cell::RefCell,
    collections::{HashMap, LinkedList},
    mem::swap,
    ops::Deref,
    rc::Rc,
};

pub use cpu::*;
#[cfg(feature = "cuda")]
mod cuda;
use crate::{OperatorTrait, Tensor, TensorF32};
use core::fmt::Debug;
#[cfg(feature = "cuda")]
pub use cuda::*;
mod buffer;
pub use buffer::*;

pub struct MemoryInfo {
    pub used: usize,
    pub free: usize,
    pub total: usize,
    pub model_parameters: usize,
}

pub trait DeviceInterface {
    fn device_buffer(&self, len: usize) -> DevBufferEnum;
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
    used: Rc<RefCell<usize>>,
    tensors_to_optimize: Rc<RefCell<Vec<Tensor>>>,
    device: Rc<dyn DeviceInterface>,
    available_buffers: Rc<RefCell<HashMap<usize, LinkedList<DevBuffer>>>>,
}

impl Debug for dyn DeviceInterface {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl DeviceInterface for Device {
    fn device_buffer(&self, len: usize) -> DevBufferEnum {
        self.device.deref().device_buffer(len)
    }

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
        self.device
            .deref()
            .sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
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
        self.device.deref().saxpy(n, alpha, x, incx, y, incy)
    }

    fn sdot(
        &self,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Result<f32, Error> {
        self.device.deref().sdot(n, x, incx, y, incy)
    }

    fn scopy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        self.device.deref().scopy(n, x, incx, y, incy)
    }

    fn sscal(&self, n: i32, alpha: f32, x: *mut f32, incx: i32) -> Result<(), Error> {
        self.device.deref().sscal(n, alpha, x, incx)
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl Device {
    pub fn new(device: &Rc<dyn DeviceInterface>) -> Self {
        Self {
            used: Default::default(),
            tensors_to_optimize: Rc::new(RefCell::new(vec![])),
            device: device.to_owned(),
            available_buffers: Default::default(),
        }
    }

    pub fn cpu() -> Self {
        let device: Rc<dyn DeviceInterface> = Rc::new(CpuDevice::default());
        Self::new(&device)
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
        let gradients = self.tensors_with_requires_grad();
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
            Ok(cublas) => {
                let device: Rc<dyn DeviceInterface> = Rc::new(cublas);
                Ok(Self::new(&device))
            }
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
        optimize: bool,
    ) -> Tensor {
        let len = rows * cols;
        let tensor = Tensor::new(
            self,
            operator,
            inputs,
            Rc::new(RefCell::new(Self::tensor_f32(&self, rows, cols, values))),
            Rc::new(RefCell::new(Self::tensor_f32(
                &self,
                rows,
                cols,
                vec![0.0; len],
            ))),
            requires_grad,
        );
        if optimize {
            self.tensors_to_optimize
                .deref()
                .borrow_mut()
                .push(tensor.clone())
        }
        tensor
    }

    pub fn tensors_with_requires_grad(&self) -> &Rc<RefCell<Vec<Tensor>>> {
        &self.tensors_to_optimize
    }

    pub fn zero_grad(&self) -> Result<(), Error> {
        let gradients: &[Tensor] = &self.tensors_with_requires_grad().deref().borrow();
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
