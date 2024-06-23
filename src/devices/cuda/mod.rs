use std::{fs::File, io::Read, sync::Arc};
pub mod slice;
pub mod stream;
#[cfg(test)]
mod tests;

use cudarc::{
    cublas::{
        sys::{cublasHandle_t, cublasOperation_t, cublasPointerMode_t, lib},
        CudaBlas,
    },
    driver::{self, CudaDevice, CudaFunction, CudaStream, DevicePtrMut, LaunchAsync, LaunchConfig},
};
use stream::CudaDeviceStream;

use crate::{
    error,
    slice::DeviceSlice,
    stream::{DeviceStream, DeviceStreamEnum},
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

    fn launch_binary_kernel(
        &self,
        module_name: &str,
        func_name: &str,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
        let n = left.len();
        let kernel = self.get_func(module_name, func_name)?;
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
                let result = unsafe {
                    kernel.launch_on_stream(
                        cuda_stream,
                        cfg,
                        (left.slice(), right.slice(), result.slice(), n),
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

    fn launch_unary_kernel(
        &self,
        module_name: &str,
        func_name: &str,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
        let kernel = self.get_func(module_name, func_name)?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let result = unsafe {
                    kernel.launch_on_stream(cuda_stream, cfg, (input.slice(), output.slice(), n))
                };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn launch_axis_kernel(
        &self,
        module_name: &str,
        func_name: &str,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
        let kernel = self.get_func(module_name, func_name)?;
        let rows = input.rows();
        let cols = input.cols();
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let result = unsafe {
                    kernel.launch_on_stream(
                        cuda_stream,
                        cfg,
                        (input.slice(), output.slice(), rows, cols),
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
            "min_kernel_module",
            &["min_kernel"],
            "./src/devices/cuda/kernels/min_kernel.cu",
        )?;

        device.load_module(
            "sigmoid_kernel_module",
            &["sigmoid_kernel"],
            "./src/devices/cuda/kernels/sigmoid_kernel.cu",
        )?;

        device.load_module(
            "gelu_kernel_module",
            &["gelu_kernel"],
            "./src/devices/cuda/kernels/gelu_kernel.cu",
        )?;

        device.load_module(
            "gelu_derivative_kernel_module",
            &["gelu_derivative_kernel"],
            "./src/devices/cuda/kernels/gelu_kernel.cu",
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

        device.load_module(
            "standardization_kernel_module",
            &["standardization_kernel"],
            "./src/devices/cuda/kernels/standardization_kernel.cu",
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
        a: &Tensor,
        lda: i32,
        b: &Tensor,
        ldb: i32,
        beta: f32,
        c: &Tensor,
        ldc: i32,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let handle = get_cublas_handle(device_stream)?;
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

        let a = a.as_ptr();
        let b = b.as_ptr();
        let c = c.as_mut_ptr();
        unsafe {
            lib().cublasSgemm_v2(
                handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            )
        }
        .result()
        .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
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
        let handle = get_cublas_handle(device_stream)?;
        let alpha = &alpha as *const f32;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        unsafe { lib().cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) }
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn dot(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
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
                let result = unsafe {
                    kernel.launch_on_stream(
                        cuda_stream,
                        cfg,
                        (left.slice(), right.slice(), result.slice(), n),
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
        let handle = get_cublas_handle(device_stream)?;
        let x = x.as_ptr();
        let x = x.wrapping_add(x_offset as usize);
        let y = y.as_mut_ptr();
        let y = y.wrapping_add(y_offset as usize);
        unsafe { lib().cublasScopy_v2(handle, n, x, x_inc, y, y_inc) }
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn scalar_mul(
        &self,
        alpha: &Tensor,
        x: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
        let n = x.len();
        let alpha = &alpha.device_slice().buffer;
        let x = &x.device_slice().buffer;
        let kernel = self.get_func("scalar_mul_kernel_module", "scalar_mul_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        match (alpha, x) {
            (DeviceSlice::CudaDevSlice(alpha), DeviceSlice::CudaDevSlice(x)) => {
                let result = unsafe {
                    kernel.launch_on_stream(cuda_stream, cfg, (n, x.slice(), alpha.slice()))
                };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn scalar_add(
        &self,
        alpha: &Tensor,
        x: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
        let n = x.len();
        let alpha = &alpha.device_slice().buffer;
        let x = &x.device_slice().buffer;
        let kernel = self.get_func("scalar_add_kernel_module", "scalar_add_kernel")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        match (alpha, x) {
            (DeviceSlice::CudaDevSlice(alpha), DeviceSlice::CudaDevSlice(x)) => {
                let result = unsafe {
                    kernel.launch_on_stream(cuda_stream, cfg, (n, x.slice(), alpha.slice()))
                };
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

    fn softmax(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_axis_kernel(
            "softmax_kernel_module",
            "softmax_kernel",
            input,
            output,
            device_stream,
        )
    }

    fn standardization(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_axis_kernel(
            "standardization_kernel_module",
            "standardization_kernel",
            input,
            output,
            device_stream,
        )
    }

    fn reduce_sum(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
        let sum_kernel = self.get_func("sum_kernel_module", "sum_kernel")?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let result = unsafe {
                    sum_kernel.launch_on_stream(
                        cuda_stream,
                        cfg,
                        (input.slice(), n, output.slice()),
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

    fn mul(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_binary_kernel(
            "mul_kernel_module",
            "mul_kernel",
            left,
            right,
            result,
            device_stream,
        )
    }

    fn sigmoid(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_unary_kernel(
            "sigmoid_kernel_module",
            "sigmoid_kernel",
            input,
            output,
            device_stream,
        )
    }

    fn gelu(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_unary_kernel(
            "gelu_kernel_module",
            "gelu_kernel",
            input,
            output,
            device_stream,
        )
    }

    fn gelu_derivative(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_unary_kernel(
            "gelu_derivative_kernel_module",
            "gelu_derivative_kernel",
            input,
            output,
            device_stream,
        )
    }

    fn div(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_binary_kernel(
            "div_kernel_module",
            "div_kernel",
            left,
            right,
            result,
            device_stream,
        )
    }

    fn min(
        &self,
        left: &Tensor,
        right: &Tensor,
        result: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_binary_kernel(
            "min_kernel_module",
            "min_kernel",
            left,
            right,
            result,
            device_stream,
        )
    }

    fn sqrt(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        self.launch_unary_kernel(
            "sqrt_kernel_module",
            "sqrt_kernel",
            input,
            output,
            device_stream,
        )
    }

    fn clip(
        &self,
        min: &Tensor,
        max: &Tensor,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
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
                    kernel.launch_on_stream(
                        cuda_stream,
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
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let cuda_stream = get_cuda_stream(device_stream)?;
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
                    kernel.launch_on_stream(
                        cuda_stream,
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

    fn reduce_sum_square(
        &self,
        expected: &Tensor,
        actual: &Tensor,
        loss: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let _cuda_stream = get_cuda_stream(device_stream)?;
        // TODO implement this in CUDA
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

    fn transpose(
        &self,
        input: &Tensor,
        output: &Tensor,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let _cuda_stream = get_cuda_stream(device_stream)?;
        // TODO implement transpose in CUDA.
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
        let cuda_stream = get_cuda_stream(device_stream)?;
        let kernel = self.get_func("bernoulli_kernel_module", "bernoulli_kernel")?;
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().buffer;
        let output = &output.device_slice().buffer;
        let rng_state = if let DeviceStreamEnum::CudaDeviceStream(stream) = &device_stream.variant {
            &stream.rng_state
        } else {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        };
        match (input, output) {
            (DeviceSlice::CudaDevSlice(input), DeviceSlice::CudaDevSlice(output)) => {
                let rng_state = rng_state;
                let result = unsafe {
                    kernel.launch_on_stream(
                        cuda_stream,
                        cfg,
                        (input.slice(), output.slice(), n, rng_state),
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

    fn stream(&self) -> Result<DeviceStreamEnum, Error> {
        let stream = self
            .dev
            .fork_default_stream()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        let rng_state = self
            .dev
            .htod_copy(vec![1337])
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        let cuda_blas =
            CudaBlas::new(self.dev.clone()).map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        // TODO uncomment
        let handle = cuda_blas.handle();

        // Set NVIDIA CUDA cublas workspace
        // See https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace
        let workspace_size_in_bytes = 32 * 1024 * 1024;
        let mut workspace = self
            .dev
            .alloc_zeros(workspace_size_in_bytes)
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        unsafe {
            lib().cublasSetWorkspace_v2(
                *handle,
                *workspace.device_ptr_mut() as *mut _,
                workspace_size_in_bytes,
            )
        }
        .result()
        .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;

        unsafe { cuda_blas.set_stream(Some(&stream)) }
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;

        // Set pointer mode.
        unsafe {
            lib().cublasSetPointerMode_v2(*handle, cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
        }
        .result()
        .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;

        let cuda_stream = CudaDeviceStream {
            device: self.dev.clone(),
            stream,
            rng_state,
            cuda_blas,
            workspace,
        };
        Ok(DeviceStreamEnum::CudaDeviceStream(cuda_stream))
    }
}

fn get_cuda_stream(device_stream: &DeviceStream) -> Result<&CudaStream, Error> {
    if let DeviceStreamEnum::CudaDeviceStream(stream) = &device_stream.variant {
        Ok(&stream.stream)
    } else {
        Err(error!(ErrorEnum::NvLaunchError))
    }
}

fn get_cublas_handle(device_stream: &DeviceStream) -> Result<cublasHandle_t, Error> {
    if let DeviceStreamEnum::CudaDeviceStream(stream) = &device_stream.variant {
        Ok(*stream.cuda_blas.handle())
    } else {
        Err(error!(ErrorEnum::UnsupportedOperation))
    }
}
