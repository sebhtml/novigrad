use std::{fs::File, io::Read, ops::Deref, sync::Arc};

mod tests;

use cudarc::{
    cublas::{
        sys::{cublasOperation_t, cublasSaxpy_v2, cublasScopy_v2, cublasSdot_v2, cublasSgemm_v2},
        CudaBlas,
    },
    driver::{self, CudaDevice, LaunchAsync, LaunchConfig},
};

use crate::{error, DevSliceEnum, DeviceInterface, Error, ErrorEnum, Tensor};

#[derive(Debug)]
pub struct CudaDev {
    cuda_blas: CudaBlas,
    pub dev: Arc<CudaDevice>,
}

impl CudaDev {
    pub fn try_default() -> Result<CudaDev, Error> {
        let dev = CudaDevice::new(0);
        let cuda_blas = dev.clone().map(|x| CudaBlas::new(x));
        match (cuda_blas, dev) {
            (Ok(Ok(cuda_blas)), Ok(dev)) => Self::try_new(cuda_blas, dev),

            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }

    pub fn try_new(cuda_blas: CudaBlas, dev: Arc<driver::CudaDevice>) -> Result<Self, Error> {
        let device = CudaDev { cuda_blas, dev };

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
            "sigmoid_module",
            &["sigmoid"],
            "./src/devices/cuda/kernels/sigmoid.cu",
        )?;

        device.load_module(
            "sqrt_kernel_module",
            &["sqrt_kernel"],
            "./src/devices/cuda/kernels/sqrt_kernel.cu",
        )?;

        device.load_module(
            "softmax_kernel_module",
            &["softmax_kernel"],
            "./src/devices/cuda/kernels/softmax.cu",
        )?;

        Ok(device)
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

impl DeviceInterface for CudaDev {
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
    ) -> Result<(), Error> {
        let handle = *self.cuda_blas.handle();
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

        let status = unsafe {
            cublasSgemm_v2(
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
    ) -> Result<(), Error> {
        let handle = *self.cuda_blas.handle();
        let alpha = &alpha as *const f32;
        let status = unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) };
        status
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn dot(&self, x: &Tensor, y: &Tensor, output: &Tensor) -> Result<(), Error> {
        let n = x.len() as i32;
        let incx = 1;
        let incy = 1;
        let x = x.as_ptr();
        let y = y.as_ptr();
        let handle = *self.cuda_blas.handle();
        let mut result: f32 = 0.0;
        let status = unsafe {
            let result = &mut result as *mut f32;
            cublasSdot_v2(handle, n, x, incx, y, incy, result)
        };
        status
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        output.set_values(vec![result])?;
        Ok(())
    }

    fn copy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        let handle = *self.cuda_blas.handle();
        let status = unsafe { cublasScopy_v2(handle, n, x, incx, y, incy) };
        status
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn scalar_mul(&self, alpha: &Tensor, x: &Tensor) -> Result<(), Error> {
        let n = x.len();
        let alpha = &alpha.device_slice().deref().borrow().buffer;
        let x = &x.device_slice().deref().borrow().buffer;
        let kernel = self
            .dev
            .get_func("scalar_mul_kernel_module", "scalar_mul_kernel")
            .unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        match (alpha, x) {
            (DevSliceEnum::CudaDevSlice(alpha), DevSliceEnum::CudaDevSlice(x)) => {
                let result = unsafe { kernel.launch(cfg, (n, x, alpha)) };
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
        let alpha = &alpha.device_slice().deref().borrow().buffer;
        let x = &x.device_slice().deref().borrow().buffer;
        let kernel = self
            .dev
            .get_func("scalar_add_kernel_module", "scalar_add_kernel")
            .unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        match (alpha, x) {
            (DevSliceEnum::CudaDevSlice(alpha), DevSliceEnum::CudaDevSlice(x)) => {
                let result = unsafe { kernel.launch(cfg, (n, x, alpha)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn slice(&self, n: i32) -> Result<DevSliceEnum, Error> {
        match self.dev.alloc_zeros(n as usize) {
            Ok(slice) => Ok(DevSliceEnum::CudaDevSlice(slice)),
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }

    fn softmax(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let kernel = self
            .dev
            .get_func("softmax_kernel_module", "softmax_kernel")
            .unwrap();
        let rows = input.rows();
        let cols = input.cols();
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().deref().borrow().buffer;
        let output = &output.device_slice().deref().borrow().buffer;
        match (input, output) {
            (DevSliceEnum::CudaDevSlice(input), DevSliceEnum::CudaDevSlice(output)) => {
                let result = unsafe { kernel.launch(cfg, (input, output, rows, cols)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }

    fn sum(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let sum_kernel = self
            .dev
            .get_func("sum_kernel_module", "sum_kernel")
            .unwrap();
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().deref().borrow().buffer;
        let output = &output.device_slice().deref().borrow().buffer;
        match (input, output) {
            (DevSliceEnum::CudaDevSlice(input), DevSliceEnum::CudaDevSlice(output)) => {
                let result = unsafe { sum_kernel.launch(cfg, (input, n, output)) };
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
        let kernel = self
            .dev
            .get_func("mul_kernel_module", "mul_kernel")
            .unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let left: &_ = &left.device_slice().deref().borrow().buffer;
        let right = &right.device_slice().deref().borrow().buffer;
        let result: &_ = &result.device_slice().deref().borrow().buffer;

        match (left, right, result) {
            (
                DevSliceEnum::CudaDevSlice(left),
                DevSliceEnum::CudaDevSlice(right),
                DevSliceEnum::CudaDevSlice(result),
            ) => {
                let result = unsafe { kernel.launch(cfg, (left, right, result, n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn sigmoid(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let kernel = self.dev.get_func("sigmoid_module", "sigmoid").unwrap();
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().deref().borrow().buffer;
        let output = &output.device_slice().deref().borrow().buffer;
        match (input, output) {
            (DevSliceEnum::CudaDevSlice(input), DevSliceEnum::CudaDevSlice(output)) => {
                let result = unsafe { kernel.launch(cfg, (input, output, n)) };
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
        let kernel = self
            .dev
            .get_func("div_kernel_module", "div_kernel")
            .unwrap();
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let left: &_ = &left.device_slice().deref().borrow().buffer;
        let right = &right.device_slice().deref().borrow().buffer;
        let result: &_ = &result.device_slice().deref().borrow().buffer;

        match (left, right, result) {
            (
                DevSliceEnum::CudaDevSlice(left),
                DevSliceEnum::CudaDevSlice(right),
                DevSliceEnum::CudaDevSlice(result),
            ) => {
                let result = unsafe { kernel.launch(cfg, (left, right, result, n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvLaunchError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvLaunchError)),
        }
    }

    fn sqrt(&self, input: &Tensor, output: &Tensor) -> Result<(), Error> {
        let kernel = self
            .dev
            .get_func("sqrt_kernel_module", "sqrt_kernel")
            .unwrap();
        let n = input.len();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let input = &input.device_slice().deref().borrow().buffer;
        let output = &output.device_slice().deref().borrow().buffer;
        match (input, output) {
            (DevSliceEnum::CudaDevSlice(input), DevSliceEnum::CudaDevSlice(output)) => {
                let result = unsafe { kernel.launch(cfg, (input, output, n)) };
                match result {
                    Ok(_) => Ok(()),
                    Err(_) => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
                }
            }
            _ => Err(error!(ErrorEnum::NvRtcLoadPtxError)),
        }
    }
}
