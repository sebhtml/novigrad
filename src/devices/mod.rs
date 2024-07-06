mod cpu;
use crate::{error, tensor::Error, tensor::ErrorEnum};
use std::mem;
use std::{
    fmt,
    ops::Deref,
    sync::{Arc, RwLock},
};
#[cfg(test)]
mod tests;

pub use cpu::*;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
use stream::{DeviceStream, DeviceStreamEnum};

use crate::{tensor::Tensor, TensorWithGrad};
pub mod slice;
pub mod stream;
use core::fmt::Debug;

use self::slice::{DevSlice, DeviceSlice};

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! new_tensor {
    ( $device:expr, $rows:expr, $cols:expr, $values:expr $(,)? ) => {
        $device.tensor($rows, $cols, $values, file!(), line!(), column!())
    };
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! new_tensor {
    ( $device:expr, $rows:expr, $cols:expr, $values:expr $(,)? ) => {
        $device.tensor($rows, $cols, $values)
    };
}

#[cfg(debug_assertions)]
#[macro_export]
macro_rules! new_tensor_with_grad {
    ( $device:expr, $rows:expr, $cols:expr, $values:expr, $inputs:expr, $requires_grad:expr, $optimize:expr $(,)? ) => {
        $device.tensor_with_grad(
            $rows,
            $cols,
            $values,
            $inputs,
            $requires_grad,
            $optimize,
            file!(),
            line!(),
            column!(),
        )
    };
}

#[cfg(not(debug_assertions))]
#[macro_export]
macro_rules! new_tensor_with_grad {
    ( $device:expr, $rows:expr, $cols:expr, $values:expr, $inputs:expr, $requires_grad:expr, $optimize:expr $(,)? ) => {
        $device.tensor_with_grad($rows, $cols, $values, $inputs, $requires_grad, $optimize)
    };
}

pub struct MemoryInfo {
    pub used: usize,
    pub free: usize,
    pub total: usize,
}

pub trait DeviceTrait {
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
        a: &Tensor,
        lda: i32,
        b: &Tensor,
        ldb: i32,
        beta: f32,
        c: &Tensor,
        ldc: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn div(
        &self,
        input1: &Tensor,
        input2: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn min(
        &self,
        input1: &Tensor,
        input2: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    /// SAXPY constant times a vector plus a vector.
    /// y = alpha * x + y
    fn axpy(
        &self,
        n: i32,
        alpha: f32,
        x: &Tensor,
        incx: i32,
        y: &Tensor,
        incy: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    /// dot performs the dot product of two vectors.
    fn dot(
        &self,
        left: &Tensor,
        right: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn clip(
        &self,
        min: &Tensor,
        max: &Tensor,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    /// SCOPY copies a vector, x, to a vector, y.
    fn copy(
        &self,
        n: i32,
        x: &Tensor,
        x_offset: i32,
        x_inc: i32,
        y: &Tensor,
        y_offset: i32,
        y_inc: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn scalar_mul(
        &self,
        alpha: &Tensor,
        x: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn scalar_add(
        &self,
        alpha: &Tensor,
        x: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn mul(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn pow(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn softmax(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn standardization(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn sigmoid(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn gelu(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn gelu_derivative(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn sqrt(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn reduce_sum(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    /// H(P, Q) = - Σ (P(i) * log(Q(i)))
    /// https://en.wikipedia.org/wiki/Entropy_(information_theory)
    fn cross_entropy_loss(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    /// RSS = Σ (y_i - f(x_i))^2
    fn reduce_sum_square(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn transpose(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    /// Allocate a slice on the device.
    fn slice(&self, n: i32) -> Result<DeviceSlice, Error>;

    fn stream(&self) -> Result<DeviceStreamEnum, Error>;
}

impl Debug for dyn DeviceTrait + Send + Sync {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

#[derive(Clone)]
pub struct Device {
    next_name: Arc<RwLock<usize>>,
    used: Arc<RwLock<usize>>,
    tensors: Arc<RwLock<Vec<Tensor>>>,
    internal_tensors: Arc<RwLock<Vec<TensorWithGrad>>>,
    parameter_tensors: Arc<RwLock<Vec<TensorWithGrad>>>,
    device: Arc<dyn DeviceTrait + Send + Sync>,
}

impl Default for Device {
    fn default() -> Self {
        #[cfg(feature = "cuda")]
        return Self::cuda().unwrap();
        #[cfg(not(feature = "cuda"))]
        return Self::cpu();
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Device")
    }
}

impl Device {
    pub fn new(device: Arc<dyn DeviceTrait + Send + Sync>) -> Self {
        Self {
            next_name: Default::default(),
            used: Default::default(),
            tensors: Default::default(),
            internal_tensors: Default::default(),
            parameter_tensors: Default::default(),
            device,
        }
    }

    pub fn new_stream(&self) -> Result<DeviceStream, Error> {
        let variant = self.stream()?;
        DeviceStream::try_new(self, variant)
    }

    pub fn copy_to(
        &self,
        x: &Tensor,
        y: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        if x.name() == y.name() {
            return Ok(());
        }
        if x.len() != y.len() {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        }
        let n = x.len() as i32;
        let (x_offset, x_inc, y_offset, y_inc) = (0, 1, 0, 1);
        self.device
            .copy(n, x, x_offset, x_inc, y, y_offset, y_inc, device_stream)
    }

    pub fn cpu() -> Self {
        Self::new(Arc::new(CpuDevice::default()))
    }

    pub fn get_memory_info(&self) -> Result<MemoryInfo, Error> {
        Ok(MemoryInfo {
            used: *self.used.read().unwrap(),
            free: 0,
            total: 0,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self, Error> {
        match CudaDev::try_default() {
            Ok(cuda) => Ok(Self::new(Arc::new(cuda))),
            Err(error) => Err(error),
        }
    }

    pub fn tensor(
        &self,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        #[cfg(debug_assertions)] file: &str,
        #[cfg(debug_assertions)] line: u32,
        #[cfg(debug_assertions)] column: u32,
    ) -> Result<Tensor, Error> {
        let name = *self.next_name.read().unwrap();
        *self.next_name.write().unwrap() += 1;
        let tensor = Tensor::new(
            name,
            rows,
            cols,
            values,
            self,
            #[cfg(debug_assertions)]
            file,
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column,
        )?;
        self.tensors.write().unwrap().push(tensor.clone());

        Ok(tensor)
    }

    pub fn tensor_with_grad(
        &self,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        inputs: &[&TensorWithGrad],
        requires_grad: bool,
        optimize: bool,
        #[cfg(debug_assertions)] file: &str,
        #[cfg(debug_assertions)] line: u32,
        #[cfg(debug_assertions)] column: u32,
    ) -> Result<TensorWithGrad, Error> {
        let len = rows * cols;
        let tensor = Self::tensor(
            self,
            rows,
            cols,
            values,
            #[cfg(debug_assertions)]
            file,
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column,
        )?;
        let gradient = if requires_grad {
            Self::tensor(
                self,
                rows,
                cols,
                vec![0.0; len],
                #[cfg(debug_assertions)]
                file,
                #[cfg(debug_assertions)]
                line,
                #[cfg(debug_assertions)]
                column,
            )?
        } else {
            Self::tensor(
                self,
                0,
                0,
                vec![],
                #[cfg(debug_assertions)]
                file,
                #[cfg(debug_assertions)]
                line,
                #[cfg(debug_assertions)]
                column,
            )?
        };
        let tensor = TensorWithGrad::new(tensor, gradient, inputs);
        if requires_grad {
            if optimize {
                self.parameter_tensors.write().unwrap().push(tensor.clone())
            } else {
                self.internal_tensors.write().unwrap().push(tensor.clone());
            }
        }

        Ok(tensor)
    }

    pub fn tensor_count(&self) -> usize {
        *self.next_name.read().unwrap()
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        for t in self.parameter_tensors.read().unwrap().iter() {
            count += t.tensor().len();
        }
        count
    }

    pub fn tensors(&self) -> impl Deref<Target = Vec<Tensor>> + '_ {
        self.tensors.read().unwrap()
    }

    pub fn internal_tensors(&self) -> impl Deref<Target = Vec<TensorWithGrad>> + '_ {
        self.internal_tensors.read().unwrap()
    }

    pub fn parameter_tensors(&self) -> impl Deref<Target = Vec<TensorWithGrad>> + '_ {
        self.parameter_tensors.read().unwrap()
    }

    pub fn buffer(&self, len: usize) -> DevSlice {
        let used: &mut usize = &mut self.used.write().unwrap();
        let bytes = len * mem::size_of::<f32>();
        *used += bytes;
        DevSlice::new(self, len)
    }
}

impl DeviceTrait for Device {
    fn gemm(
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
        c: &Tensor,
        ldc: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.gemm(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            device_stream,
        )
    }

    fn dot(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        if *left.size() != *right.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        if &result.size() as &[usize] != &[1, 1] {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        self.device.dot(left, right, result, device_stream)
    }

    fn copy(
        &self,
        n: i32,
        x: &Tensor,
        x_offset: i32,
        x_inc: i32,
        y: &Tensor,
        y_offset: i32,
        y_inc: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device
            .copy(n, x, x_offset, x_inc, y, y_offset, y_inc, device_stream)
    }

    fn axpy(
        &self,
        n: i32,
        alpha: f32,
        x: &Tensor,
        incx: i32,
        y: &Tensor,
        incy: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.axpy(n, alpha, x, incx, y, incy, device_stream)
    }

    fn scalar_mul(
        &self,
        alpha: &Tensor,
        x: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.scalar_mul(alpha, x, device_stream)
    }

    fn scalar_add(
        &self,
        alpha: &Tensor,
        x: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.scalar_add(alpha, x, device_stream)
    }

    fn softmax(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.softmax(input, output, device_stream)
    }

    fn reduce_sum(
        &self,
        x: &Tensor,
        y: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        if &y.size() as &[usize] != &[1, 1] {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        self.device.reduce_sum(x, y, device_stream)
    }

    fn mul(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.mul(left, right, result, device_stream)
    }

    fn pow(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.pow(left, right, result, device_stream)
    }

    fn sigmoid(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.sigmoid(input, output, device_stream)
    }

    fn gelu(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.gelu(input, output, device_stream)
    }

    fn gelu_derivative(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.gelu_derivative(input, output, device_stream)
    }

    fn sqrt(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.sqrt(input, output, device_stream)
    }

    fn div(
        &self,
        input1: &Tensor,
        input2: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.div(input1, input2, output, device_stream)
    }

    fn min(
        &self,
        input1: &Tensor,
        input2: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.min(input1, input2, output, device_stream)
    }

    fn clip(
        &self,
        min: &Tensor,
        max: &Tensor,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.clip(min, max, input, output, device_stream)
    }

    fn cross_entropy_loss(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device
            .cross_entropy_loss(expected, actual, loss, device_stream)
    }

    fn reduce_sum_square(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device
            .reduce_sum_square(expected, actual, loss, device_stream)
    }

    fn transpose(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.transpose(input, output, device_stream)
    }

    fn slice(&self, n: i32) -> Result<DeviceSlice, Error> {
        self.device.slice(n)
    }

    fn stream(&self) -> Result<DeviceStreamEnum, Error> {
        self.device.stream()
    }

    fn standardization(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.standardization(input, output, device_stream)
    }
}
