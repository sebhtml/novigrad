use std::{ffi::c_void, fs::File, io::Read, sync::Arc};

mod tests;

use cudarc::{
    cublas::{
        sys::{
            cublasOperation_t, cublasSaxpy_v2, cublasScopy_v2, cublasSdot_v2, cublasSgemmEx,
            cublasSscal_v2, cudaDataType,
        },
        CudaBlas,
    },
    driver,
};

use crate::{DevBufferEnum, DeviceInterface, Error, ErrorEnum};

#[derive(Debug)]
pub struct CudaDevice {
    cuda_blas: CudaBlas,
    pub dev: Arc<driver::CudaDevice>,
}

impl CudaDevice {
    pub fn try_default() -> Result<CudaDevice, Error> {
        let dev = cudarc::driver::CudaDevice::new(0);
        let cuda_blas = dev.clone().map(|x| CudaBlas::new(x));
        match (cuda_blas, dev) {
            (Ok(Ok(cuda_blas)), Ok(dev)) => Self::try_new(cuda_blas, dev),

            _ => Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::UnsupportedOperation,
            )),
        }
    }

    pub fn try_new(cuda_blas: CudaBlas, dev: Arc<driver::CudaDevice>) -> Result<Self, Error> {
        let device = CudaDevice { cuda_blas, dev };

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
            .map_err(|_| Error::new(file!(), line!(), column!(), ErrorEnum::InputOutputError))?
            .read_to_string(&mut cuda_code)
            .map_err(|_| Error::new(file!(), line!(), column!(), ErrorEnum::InputOutputError))?;
        let ptx = cudarc::nvrtc::compile_ptx(cuda_code).map_err(|err| {
            Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::NvRtcCompilePtxError(err),
            )
        })?;

        self.dev
            .load_ptx(ptx, module_name, func_names)
            .map_err(|_| Error::new(file!(), line!(), column!(), ErrorEnum::NvRtcLoadPtxError))?;
        Ok(())
    }
}

impl DeviceInterface for CudaDevice {
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
        let handle = *self.cuda_blas.handle();
        let transa = match transa {
            false => cublasOperation_t::CUBLAS_OP_N,
            true => cublasOperation_t::CUBLAS_OP_T,
        };
        let transb = match transb {
            false => cublasOperation_t::CUBLAS_OP_N,
            true => cublasOperation_t::CUBLAS_OP_T,
        };
        let a = a as *const c_void;
        let b = b as *const c_void;
        let c = c as *mut c_void;
        let a_type = cudaDataType::CUDA_R_32F;
        let b_type = cudaDataType::CUDA_R_32F;
        let c_type = cudaDataType::CUDA_R_32F;
        let alpha = &alpha as *const f32;
        let beta = &beta as *const f32;

        let status = unsafe {
            cublasSgemmEx(
                handle, transa, transb, m, n, k, alpha, a, a_type, lda, b, b_type, ldb, beta, c,
                c_type, ldc,
            )
        };
        status
            .result()
            .map_err(|_| Error::new(file!(), line!(), column!(), ErrorEnum::UnsupportedOperation))
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
        let handle = *self.cuda_blas.handle();
        let alpha = &alpha as *const f32;
        let status = unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) };
        status
            .result()
            .map_err(|_| Error::new(file!(), line!(), column!(), ErrorEnum::UnsupportedOperation))
    }

    fn sdot(
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
        status.result().map_err(|_| {
            Error::new(file!(), line!(), column!(), ErrorEnum::UnsupportedOperation)
        })?;
        Ok(result)
    }

    fn scopy(&self, n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        let handle = *self.cuda_blas.handle();
        let status = unsafe { cublasScopy_v2(handle, n, x, incx, y, incy) };
        status
            .result()
            .map_err(|_| Error::new(file!(), line!(), column!(), ErrorEnum::UnsupportedOperation))
    }

    fn scalar_mul(&self, n: i32, alpha: *const f32, x: *mut f32, incx: i32) -> Result<(), Error> {
        let handle = *self.cuda_blas.handle();
        let status = unsafe { cublasSscal_v2(handle, n, alpha, x, incx) };
        status
            .result()
            .map_err(|_| Error::new(file!(), line!(), column!(), ErrorEnum::UnsupportedOperation))
    }

    fn slice(&self, n: i32) -> Result<DevBufferEnum, Error> {
        match self.dev.alloc_zeros(n as usize) {
            Ok(slice) => Ok(DevBufferEnum::CudaBuffer(slice)),
            _ => Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::UnsupportedOperation,
            )),
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

    fn sum(&self, _n: i32, _x: *const f32, _incx: i32, _y: *mut f32) -> Result<(), Error> {
        todo!()
    }
}
