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
            "mul_kernel_module",
            &["mul_kernel"],
            "./src/devices/cuda/kernels/mul_kernel.cu",
        )?;

        device.load_module(
            "sigmoid_module",
            &["sigmoid"],
            "./src/devices/cuda/kernels/sigmoid.cu",
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
    fn gemm(
        &self,
        transa: bool,
        transb: bool,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Tensor,
        a: &Tensor,
        lda: i32,
        b: &Tensor,
        ldb: i32,
        beta: &Tensor,
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
        // When they are on the host, it's fine though.
        //let alpha = alpha.as_ptr();
        let alpha = alpha.get_values()?;
        let alpha = alpha[0];
        let alpha = &alpha;
        let beta = beta.get_values()?;
        let beta = beta[0];
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

    fn dot(
        &self,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Result<f32, Error> {
        let handle = *self.cuda_blas.handle();
        let mut result: f32 = 0.0;
        let status = unsafe {
            let result = &mut result as *mut f32;
            cublasSdot_v2(handle, n, x, incx, y, incy, result)
        };
        status
            .result()
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
        Ok(result)
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

    fn slice(&self, n: i32) -> Result<DevSliceEnum, Error> {
        match self.dev.alloc_zeros(n as usize) {
            Ok(slice) => Ok(DevSliceEnum::CudaDevSlice(slice)),
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }

    fn softmax(
        &self,
        _rows: i32,
        _cols: i32,
        _input: *const f32,
        _output: *mut f32,
    ) -> Result<(), Error> {
        todo!()
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
}
