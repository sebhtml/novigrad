use std::{fs::File, io::Read, sync::Arc};
pub mod slice;
pub mod stream;
#[cfg(test)]
mod tests;

use cudarc::{
    cublas::{
        sys::{cublasOperation_t, lib},
        CudaBlas,
    },
    driver::{self, CudaDevice, CudaFunction, LaunchAsync, LaunchConfig},
};
use stream::CudaDeviceStream;

use crate::{
    error,
    slice::DeviceSlice,
    stream::DeviceStream,
    tensor::{Error, ErrorEnum, Tensor},
    DeviceTrait, EPSILON,
};

use self::slice::CudaDevSlice;

#[derive(Debug)]
pub struct CudaDev {
    pub dev: Arc<CudaDevice>,
}

impl CudaDev {
    pub fn try_default() -> Result<CudaDev, Error> {
        let dev = CudaDevice::new(0);
        match dev {
            Ok(dev) => Self::try_new(dev),

            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }

    pub fn try_new(dev: Arc<driver::CudaDevice>) -> Result<Self, Error> {
        let device = CudaDev { dev };

        device.load_module(
            "sin_kernel_module",
            &["sin_kernel"],
            "./src/devices/cuda/kernels/sin_kernel.cu",
        )?;

        device.load_module(
            "sum_kernel_module",
            &["sum_kernel"],
            "./src/devices/cuda/kernels/sum_kernel.cu",
        )?;

        device.load_module(
            "dot_kernel_module",
            &["dot_kernel"],
            "./src/devices/cuda/kernels/dot_kernel.cu",
        )?;

        device.load_module(
            "cross_entropy_loss_kernel_module",
            &["cross_entropy_loss_kernel"],
            "./src/devices/cuda/kernels/cross_entropy_loss_kernel.cu",
        )?;

        device.load_module(
            "scalar_mul_kernel_module",
            &["scalar_mul_kernel"],
            "./src/devices/cuda/kernels/scalar_mul_kernel.cu",
        )?;

        device.load_module(
            "scalar_add_kernel_module",
            &["scalar_add_kernel"],
            "./src/devices/cuda/kernels/scalar_add_kernel.cu",
        )?;

        device.load_module(
            "mul_kernel_module",
            &["mul_kernel"],
            "./src/devices/cuda/kernels/mul_kernel.cu",
        )?;

        device.load_module(
            "div_kernel_module",
            &["div_kernel"],
            "./src/devices/cuda/kernels/div_kernel.cu",
        )?;

        device.load_module(
            "sigmoid_kernel_module",
            &["sigmoid_kernel"],
            "./src/devices/cuda/kernels/sigmoid_kernel.cu",
        )?;

        device.load_module(
            "bernoulli_kernel_module",
            &["bernoulli_kernel"],
            "./src/devices/cuda/kernels/bernoulli_kernel.cu",
        )?;

        device.load_module(
            "sqrt_kernel_module",
            &["sqrt_kernel"],
            "./src/devices/cuda/kernels/sqrt_kernel.cu",
        )?;

        device.load_module(
            "clip_kernel_module",
            &["clip_kernel"],
            "./src/devices/cuda/kernels/clip_kernel.cu",
        )?;

        device.load_module(
            "softmax_kernel_module",
            &["softmax_kernel"],
            "./src/devices/cuda/kernels/softmax_kernel.cu",
        )?;

        Ok(device)
    }

    fn get_func(&self, module_name: &str, func_name: &str) -> Result<CudaFunction, Error> {
        let kernel = self
            .dev
            .get_func(module_name, func_name)
            .ok_or(error!(ErrorEnum::NvGetFuncError(func_name.into())))?;
        Ok(kernel)
    }

    fn load_module(
        &self,
        module_name: &str,
        func_names: &[&'static str],
        src_file_path: &str,
    ) -> Result<(), Error> {
        let mut cuda_code = String::default();
        File::open(src_file_path)
            .map_err(|_| error!(ErrorEnum::InputOutputError))?
            .read_to_string(&mut cuda_code)
            .map_err(|_| error!(ErrorEnum::InputOutputError))?;
        let ptx = cudarc::nvrtc::compile_ptx(cuda_code)
            .map_err(|err| error!(ErrorEnum::NvRtcCompilePtxError(err)))?;

        self.dev
            .load_ptx(ptx, module_name, func_names)
            .map_err(|_| error!(ErrorEnum::NvRtcLoadPtxError))?;
        Ok(())
    }
}

impl DeviceTrait for CudaDev {
    /// On the web page
    /// https://docs.nvidia.com/cuda/cublas/#cublas-level-3-function-reference
    ///
    /// It says that the Param alpha Memory can be on host or device.
    /// But if I put it on device, libcublas.so does a SIGSEGV ! (a Segmentation fault)
    ///
    /// LOL.
    ///
    /// In https://github.com/coreylowman/dfdx, alpha is stored on host Memory.
    /// So this is done like that here too since the NVIDIA SGEMM documentation is bad.
    ///
    /// Maybe alpha can be on device Memory for a
    /// "NVIDIA H100 Tensor Core GPU"
    /// but not a
    /// "NVIDIA GeForce RTX™ 4060 4060"
    /// .
    /// Who knows...
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
        let handle = if let DeviceStream::CudaDeviceStream(stream) = device_stream {
            *stream.cuda_blas.handle()
        } else {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        };
        let transa = match transa {
            false => cublasOperation_t::CUBLAS_OP_N,
            true => cublasOperation_t::CUBLAS_OP_T,
        };
        let transb = match transb {
            false => cublasOperation_t::CUBLAS_OP_N,
            true => cublasOperation_t::CUBLAS_OP_T,
        };
        // TODO cublas does a segmentation fault when alpha and beta are on the GPU.
        // But the documentation says that alpha can be on the device:
        // https://docs.nvidia.com/cuda/cublas/#cublas-level-3-function-reference
        // When they are on the host, it's fine though.
        let alpha = &alpha;
        let beta = &beta;

        let status = unsafe {
            lib().cublasSgemm_v2(
                handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            )
        };
        status
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
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
        let handle = if let DeviceStream::CudaDeviceStream(stream) = device_stream {
            *stream.cuda_blas.handle()
        } else {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        };
        let alpha = &alpha as *const f32;
        let status = unsafe { lib().cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) };
        status
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn dot(&self, left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error> {
        let n = left.len();
        let kernel = self.get_func("dot_kernel_module", "dot_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let left = &left.device_slice().buffer;
        let right = &right.device_slice().buffer;
        let result = &result.device_slice().buffer;

        match (left, right, result) {
            (
                DeviceSlice::CudaDevSlice(left),
                DeviceSlice::CudaDevSlice(right),
                DeviceSlice::CudaDevSlice(result),
            ) => {
                let result =
                    unsafe { kernel.launch(cfg, (left.slice(), right.slice(), result.slice(), n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
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
        let handle = if let DeviceStream::CudaDeviceStream(stream) = device_stream {
            *stream.cuda_blas.handle()
        } else {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        };
        let status = unsafe { lib().cublasScopy_v2(handle, n, x, incx, y, incy) };
        status
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn scalar_mul(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error> {
        let n = x.len();
        let alpha = &alpha.device_slice().buffer;
        let x = &x.device_slice().buffer;
        let kernel = self.get_func("scalar_mul_kernel_module", "scalar_mul_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        match (alpha, x) {
            (DeviceSlice::CudaDevSlice(alpha), DeviceSlice::CudaDevSlice(x)) => {
                let result = unsafe { kernel.launch(cfg, (n, x.slice(), alpha.slice())) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn scalar_add(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error> {
        let n = x.len();
        let alpha = &alpha.device_slice().buffer;
        let x = &x.device_slice().buffer;
        let kernel = self.get_func("scalar_add_kernel_module", "scalar_add_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        match (alpha, x) {
            (DeviceSlice::CudaDevSlice(alpha), DeviceSlice::CudaDevSlice(x)) => {
                let result = unsafe { kernel.launch(cfg, (n, x.slice(), alpha.slice())) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn slice(&self, n: i32) -> Result<DeviceSlice, Error> {
        match self.dev.alloc_zeros(n as usize) {
            Ok(slice) => Ok(DeviceSlice::CudaDevSlice(CudaDevSlice::new(slice))),
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }

    fn softmax(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let kernel = self.get_func("softmax_kernel_module", "softmax_kernel")?;
        let rows = input.rows();
        let cols = input.cols();
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let result =
                    unsafe { kernel.launch(cfg, (input.slice(), output.slice(), rows, cols)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn sum(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let sum_kernel = self.get_func("sum_kernel_module", "sum_kernel")?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let result = unsafe { sum_kernel.launch(cfg, (input.slice(), n, output.slice())) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn mul(&self, left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error> {
        let n = left.len();
        let kernel = self.get_func("mul_kernel_module", "mul_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let left = &left.device_slice().buffer;
        let right = &right.device_slice().buffer;
        let result = &result.device_slice().buffer;

        match (left, right, result) {
            (
                DeviceSlice::CudaDevSlice(left),
                DeviceSlice::CudaDevSlice(right),
                DeviceSlice::CudaDevSlice(result),
            ) => {
                let result =
                    unsafe { kernel.launch(cfg, (left.slice(), right.slice(), result.slice(), n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn sigmoid(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let kernel = self.get_func("sigmoid_kernel_module", "sigmoid_kernel")?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let result = unsafe { kernel.launch(cfg, (input.slice(), output.slice(), n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn div(&self, left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error> {
        let n = left.len();
        let kernel = self.get_func("div_kernel_module", "div_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let left = &left.device_slice().buffer;
        let right = &right.device_slice().buffer;
        let result = &result.device_slice().buffer;

        match (left, right, result) {
            (
                DeviceSlice::CudaDevSlice(left),
                DeviceSlice::CudaDevSlice(right),
                DeviceSlice::CudaDevSlice(result),
            ) => {
                let result =
                    unsafe { kernel.launch(cfg, (left.slice(), right.slice(), result.slice(), n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn sqrt(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let kernel = self.get_func("sqrt_kernel_module", "sqrt_kernel")?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let result = unsafe { kernel.launch(cfg, (input.slice(), output.slice(), n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn clip(
        &self,
        min: &Tensor,
        max: &Tensor,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<(), Error> {
        let kernel = self.get_func("clip_kernel_module", "clip_kernel")?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let min = &min.device_slice().buffer;
        let max = &max.device_slice().buffer;
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (min, max, input, output) {
            (
                DeviceSlice::CudaDevSlice(min),
                DeviceSlice::CudaDevSlice(max),
                DeviceSlice::CudaDevSlice(input),
                DeviceSlice::CudaDevSlice(output),
            ) => {
                let result = unsafe {
                    kernel.launch(
                        cfg,
                        (min.slice(), max.slice(), input.slice(), output.slice(), n),
                    )
                };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn cross_entropy_loss(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
    ) -> Result<(), Error> {
        let n = expected.len();
        let kernel = self.get_func(
            "cross_entropy_loss_kernel_module",
            "cross_entropy_loss_kernel",
        )?;
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let expected = &expected.device_slice().buffer;
        let actual = &actual.device_slice().buffer;
        let loss = &loss.device_slice().buffer;

        match (expected, actual, loss) {
            (
                DeviceSlice::CudaDevSlice(expected),
                DeviceSlice::CudaDevSlice(actual),
                DeviceSlice::CudaDevSlice(loss),
            ) => {
                let result = unsafe {
                    kernel.launch(
                        cfg,
                        (expected.slice(), actual.slice(), loss.slice(), n, EPSILON),
                    )
                };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn reduce_square_sum(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
    ) -> Result<(), Error> {
        if *expected.size() != *actual.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        let expected_values = expected.get_values()?;
        let actual_values = actual.get_values()?;
        let mut loss_value = 0.0;
        for i in 0..expected_values.len() {
            let expected = expected_values[i];
            let actual = actual_values[i];
            let diff = expected - actual;
            loss_value += diff * diff;
        }

        loss.set_values(vec![loss_value; 1])?;
        Ok(())
    }

    fn transpose(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let self_values = input.get_values()?;
        let mut other_values = output.get_values()?;
        let rows = input.rows();
        let cols = input.cols();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let value = self_values[input.index(row, col)];
                other_values[output.index(col, row)] = value;
                col += 1;
            }
            row += 1;
        }
        output.set_values(other_values)
    }

    fn bernoulli(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let kernel = self.get_func("bernoulli_kernel_module", "bernoulli_kernel")?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        let rng_state = if let DeviceStream::CudaDeviceStream(stream) = device_stream {
            &stream.rng_state
        } else {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        };
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let rng_state = rng_state;
                let result =
                    unsafe { kernel.launch(cfg, (input.slice(), output.slice(), n, rng_state)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn stream(&self) -> Result<DeviceStream, Error> {
        let rng_state = self
            .dev
            .htod_copy(vec![1337])
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        let cuda_blas =
            CudaBlas::new(self.dev.clone()).map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        match self.dev.fork_default_stream() {
            Ok(stream) => {
                unsafe { cuda_blas.set_stream(Some(&stream)) }
                    .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
                let cuda_stream = CudaDeviceStream {
                    stream,
                    rng_state,
                    cuda_blas,
                };
                Ok(DeviceStream::CudaDeviceStream(cuda_stream))
            }
            Err(_) => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }
}
