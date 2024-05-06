use std::{ffi::c_void, sync::Arc};

mod tests;

use cudarc::{
    cublas::{
        result::create_handle,
        sys::{
            cublasHandle_t, cublasOperation_t, cublasSaxpy_v2, cublasScopy_v2, cublasSdot_v2,
            cublasSgemmEx, cublasSscal_v2, cudaDataType,
        },
    },
    driver::{self},
};

use crate::{DeviceInterface, Error, TensorF32};

#[derive(Debug)]
pub struct CudaDevice {
    handle: cublasHandle_t,
    _dev: Arc<driver::CudaDevice>,
}

impl CudaDevice {
    pub fn try_default() -> Result<CudaDevice, Error> {
        let handle = create_handle();
        let dev = cudarc::driver::CudaDevice::new(0);
        match (handle, dev) {
            (Ok(handle), Ok(dev)) => Ok(CudaDevice { handle, _dev: dev }),
            _ => Err(Error::UnsupportedOperation),
        }
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
        let handle = self.handle;
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
        status.result().map_err(|_| Error::UnsupportedOperation)
    }

    fn saxpy(
        &self,
        n: i32,
        alpha: f32,
        x: &TensorF32,
        incx: i32,
        y: &mut TensorF32,
        incy: i32,
    ) -> Result<(), Error> {
        let handle = self.handle;
        let alpha = &alpha as *const f32;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        let status = unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) };
        status.result().map_err(|_| Error::UnsupportedOperation)
    }

    fn sdot(
        &self,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> Result<f32, Error> {
        let handle = self.handle;
        let mut result: f32 = 0.0;
        let status = unsafe {
            let result = &mut result as *mut f32;
            cublasSdot_v2(handle, n, x, incx, y, incy, result)
        };
        status.result().map_err(|_| Error::UnsupportedOperation)?;
        Ok(result)
    }

    fn scopy(
        &self,
        n: i32,
        x: &TensorF32,
        incx: i32,
        y: &mut TensorF32,
        incy: i32,
    ) -> Result<(), Error> {
        let handle = self.handle;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        let status = unsafe { cublasScopy_v2(handle, n, x, incx, y, incy) };
        status.result().map_err(|_| Error::UnsupportedOperation)
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut TensorF32, incx: i32) -> Result<(), Error> {
        let handle = self.handle;
        let x = x.as_mut_ptr();
        let alpha = &alpha as *const f32;
        let status = unsafe { cublasSscal_v2(handle, n, alpha, x, incx) };
        status.result().map_err(|_| Error::UnsupportedOperation)
    }
}
