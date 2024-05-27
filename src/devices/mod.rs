mod cpu;
use crate::{error, Error, ErrorEnum};
use std::{
    cell::RefCell,
    collections::{HashMap, LinkedList},
    mem::swap,
    ops::Deref,
    rc::Rc,
};
#[cfg(test)]
mod tests;

pub use cpu::*;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

use crate::{Tensor, TensorWithGrad};
pub mod slice;
use core::fmt::Debug;

use self::slice::{DevSlice, DevSliceEnum};

pub struct MemoryInfo {
    pub used: usize,
    pub free: usize,
    pub total: usize,
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
    fn gemm(
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

    fn div(&self, input1: &Tensor, input2: &Tensor, output: &Tensor) -> Result<(), Error>;

    /// SAXPY constant times a vector plus a vector.
    /// y = alpha * x + y
    fn axpy(
        &self,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> Result<(), Error>;

    /// dot performs the dot product of two vectors.
    fn dot(&self, left: &Tensor, right: &Tensor, output: &Tensor) -> Result<(), Error>;

    fn clip(
        &self,
        min: &Tensor,
        max: &Tensor,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<(), Error>;

    /// SCOPY copies a vector, x, to a vector, y.
    fn copy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error>;

    fn scalar_mul(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error>;

    fn scalar_add(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error>;

    fn mul(&self, left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error>;

    /// Allocate a slice on the device.
    fn slice(&self, n: i32) -> Result<DevSliceEnum, Error>;

    fn softmax(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    fn sigmoid(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    fn sqrt(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    fn sum(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    /// H(P, Q) = - Σ (P(i) * log(Q(i)))
    fn cross_entropy_loss(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
    ) -> Result<(), Error>;

    /// RSS = Σ (y_i - f(x_i))^2
    fn reduce_square_sum(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
    ) -> Result<(), Error>;

    fn transpose(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;
}

impl Debug for dyn DeviceInterface {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Device {
    next_name: Rc<RefCell<usize>>,
    used: Rc<RefCell<usize>>,
    tensors_to_optimize: Rc<RefCell<Vec<TensorWithGrad>>>,
    device: Rc<dyn DeviceInterface>,
    available_buffers: Rc<RefCell<HashMap<usize, LinkedList<DevSlice>>>>,
}

impl Default for Device {
    fn default() -> Self {
        #[cfg(feature = "cuda")]
        return Self::cuda().unwrap();
        #[cfg(not(feature = "cuda"))]
        return Self::cpu();
    }
}

impl Device {
    pub fn new(device: Rc<dyn DeviceInterface>) -> Self {
        Self {
            next_name: Default::default(),
            used: Default::default(),
            tensors_to_optimize: Rc::new(RefCell::new(vec![])),
            device,
            available_buffers: Default::default(),
        }
    }

    pub fn cpu() -> Self {
        Self::new(Rc::new(CpuDevice::default()))
    }

    pub fn recycle(&self, len: usize, buffer: &mut DevSlice) {
        let mut recycled_buffer = DevSlice::new(self, 0);
        swap(&mut recycled_buffer, buffer);

        let available_buffers: &mut HashMap<_, _> =
            &mut self.available_buffers.deref().borrow_mut();
        let entry = available_buffers.entry(len);
        entry.or_default().push_back(recycled_buffer)
    }

    pub fn get_memory_info(&self) -> Result<MemoryInfo, Error> {
        Ok(MemoryInfo {
            used: *self.used.deref().borrow(),
            free: 0,
            total: 0,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self, Error> {
        match CudaDev::try_default() {
            Ok(cuda) => Ok(Self::new(Rc::new(cuda))),
            Err(error) => Err(error),
        }
    }

    pub fn tensor(&self, rows: usize, cols: usize, values: Vec<f32>) -> Result<Tensor, Error> {
        let name = *self.next_name.deref().borrow();
        *self.next_name.deref().borrow_mut() += 1;
        Tensor::new(name, rows, cols, values, self)
    }

    pub fn tensor_with_grad(
        &self,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        inputs: &[&TensorWithGrad],
        requires_grad: bool,
        optimize: bool,
    ) -> Result<TensorWithGrad, Error> {
        let len = rows * cols;
        let tensor = Self::tensor(&self, rows, cols, values)?;
        let gradient = if requires_grad {
            Self::tensor(&self, rows, cols, vec![0.0; len])?
        } else {
            Self::tensor(&self, 0, 0, vec![])?
        };
        let tensor = TensorWithGrad::new(tensor, gradient, inputs);
        if optimize {
            self.tensors_to_optimize
                .deref()
                .borrow_mut()
                .push(tensor.clone())
        }
        Ok(tensor)
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

    pub fn tensors_to_optimize(&self) -> &Rc<RefCell<Vec<TensorWithGrad>>> {
        &self.tensors_to_optimize
    }

    pub fn buffer(&self, len: usize) -> DevSlice {
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
                DevSlice::new(self, len)
            }
        }
    }
}

impl DeviceInterface for Device {
    fn gemm(
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
            .gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    fn dot(&self, left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error> {
        if left.size() != right.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        if &result.size().deref().borrow() as &[usize] != &[1, 1] {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        self.device.dot(left, right, result)
    }

    fn copy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        self.device.copy(n, x, incx, y, incy)
    }

    fn axpy(
        &self,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> Result<(), Error> {
        self.device.axpy(n, alpha, x, incx, y, incy)
    }

    fn scalar_mul(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error> {
        self.device.scalar_mul(alpha, x)
    }

    fn scalar_add(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error> {
        self.device.scalar_add(alpha, x)
    }

    fn slice(&self, n: i32) -> Result<DevSliceEnum, Error> {
        self.device.slice(n)
    }

    fn softmax(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.softmax(input, output)
    }

    fn sum(&self, x: &Tensor, y: &Tensor) -> Result<(), Error> {
        if &y.size().deref().borrow() as &[usize] != &[1, 1] {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        self.device.sum(x, y)
    }

    fn mul(&self, left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error> {
        self.device.mul(left, right, result)
    }

    fn sigmoid(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.sigmoid(input, output)
    }

    fn sqrt(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.sqrt(input, output)
    }

    fn div(&self, input1: &Tensor, input2: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.div(input1, input2, output)
    }

    fn clip(
        &self,
        min: &Tensor,
        max: &Tensor,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<(), Error> {
        self.device.clip(min, max, input, output)
    }

    fn cross_entropy_loss(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
    ) -> Result<(), Error> {
        self.device.cross_entropy_loss(expected, actual, loss)
    }

    fn reduce_square_sum(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
    ) -> Result<(), Error> {
        self.device.reduce_square_sum(expected, actual, loss)
    }

    fn transpose(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.transpose(input, output)
    }
}
