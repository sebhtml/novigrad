mod cpu;
use crate::{error, tensor::Error, tensor::ErrorEnum};
use std::{
    collections::{HashMap, LinkedList},
    mem::swap,
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
use stream::DeviceStream;

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
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
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
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn scalar_mul(
        &self,
        alpha: &Tensor,
        x: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn scalar_add(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error>;

    fn mul(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn softmax(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    fn sigmoid(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    fn bernoulli(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;

    fn sqrt(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    fn sum(&self, input: &Tensor, output: &Tensor) -> Result<(), Error>;

    /// H(P, Q) = - Σ (P(i) * log(Q(i)))
    /// https://en.wikipedia.org/wiki/Entropy_(information_theory)
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

    /// Allocate a slice on the device.
    fn slice(&self, n: i32) -> Result<DeviceSlice, Error>;

    fn stream(&self) -> Result<DeviceStream, Error>;
}

impl Debug for dyn DeviceTrait + Send + Sync {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Device {
    next_name: Arc<RwLock<usize>>,
    used: Arc<RwLock<usize>>,
    tensors: Arc<RwLock<Vec<Tensor>>>,
    tensors_to_optimize: Arc<RwLock<Vec<TensorWithGrad>>>,
    device: Arc<dyn DeviceTrait + Send + Sync>,
    available_buffers: Arc<RwLock<HashMap<usize, LinkedList<DevSlice>>>>,
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
    pub fn new(device: Arc<dyn DeviceTrait + Send + Sync>) -> Self {
        Self {
            next_name: Default::default(),
            used: Default::default(),
            tensors: Default::default(),
            tensors_to_optimize: Default::default(),
            device,
            available_buffers: Default::default(),
        }
    }

    pub fn cpu() -> Self {
        Self::new(Arc::new(CpuDevice::default()))
    }

    pub fn recycle(&self, len: usize, buffer: &mut DevSlice) {
        let mut recycled_buffer = DevSlice::new(self, 0);
        swap(&mut recycled_buffer, buffer);

        let available_buffers: &mut HashMap<_, _> = &mut self.available_buffers.write().unwrap();
        let entry = available_buffers.entry(len);
        entry.or_default().push_back(recycled_buffer)
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
            &self,
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
                &self,
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
                &self,
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
        if optimize {
            self.tensors_to_optimize
                .write()
                .unwrap()
                .push(tensor.clone())
        }
        Ok(tensor)
    }

    pub fn tensor_count(&self) -> usize {
        *self.next_name.read().unwrap()
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        for t in self.tensors_to_optimize.read().unwrap().iter() {
            count += t.tensor().len();
        }
        count
    }

    pub fn tensors(&self) -> impl Deref<Target = Vec<Tensor>> + '_ {
        self.tensors.read().unwrap()
    }

    pub fn tensors_to_optimize(&self) -> impl Deref<Target = Vec<TensorWithGrad>> + '_ {
        self.tensors_to_optimize.read().unwrap()
    }

    pub fn buffer(&self, len: usize) -> DevSlice {
        let recycled = self
            .available_buffers
            .write()
            .unwrap()
            .get_mut(&len)
            .map(|x| x.pop_back())
            .flatten();
        match recycled {
            Some(buffer) => {
                //println!("Recycled buffer with length {}", len);
                buffer
            }
            None => {
                let used: &mut usize = &mut self.used.write().unwrap();
                *used += len;
                DevSlice::new(self, len)
            }
        }
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
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
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
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.copy(n, x, incx, y, incy, device_stream)
    }

    fn axpy(
        &self,
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
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

    fn scalar_add(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error> {
        self.device.scalar_add(alpha, x)
    }

    fn slice(&self, n: i32) -> Result<DeviceSlice, Error> {
        self.device.slice(n)
    }

    fn softmax(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.softmax(input, output)
    }

    fn sum(&self, x: &Tensor, y: &Tensor) -> Result<(), Error> {
        if &y.size() as &[usize] != &[1, 1] {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        self.device.sum(x, y)
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

    fn sigmoid(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.sigmoid(input, output)
    }

    fn sqrt(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        self.device.sqrt(input, output)
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

    fn bernoulli(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.device.bernoulli(input, output, device_stream)
    }

    fn stream(&self) -> Result<DeviceStream, Error> {
        self.device.stream()
    }
}
