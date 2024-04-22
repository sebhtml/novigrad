use std::{ffi::c_void, sync::Arc};

mod tests;

use cudarc::{
    cublas::{
        result::create_handle,
        sys::{
            cublasHandle_t, cublasOperation_t, cublasSaxpy_v2, cublasScopy_v2, cublasSdot_v2,
            cublasSgemmEx, cublasSscal_v2, cublasStatus_t, cudaDataType,
        },
    },
    driver,
};

use crate::{DeviceInterface, Error, Transpose};

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

impl Into<cublasOperation_t> for Transpose {
    fn into(self) -> cublasOperation_t {
        match self {
            Transpose::None => cublasOperation_t::CUBLAS_OP_N,
            Transpose::Ordinary => cublasOperation_t::CUBLAS_OP_T,
            Transpose::Conjugate => cublasOperation_t::CUBLAS_OP_C,
        }
    }
}

impl DeviceInterface for CudaDevice {
    // TODO return Result
    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    ) {
        let handle = self.handle;
        let transa = transa.into();
        let transb = transb.into();
        let a = a.as_ptr() as *const c_void;
        let b = b.as_ptr() as *const c_void;
        let c = c.as_mut_ptr() as *mut c_void;
        let a_type = cudaDataType::CUDA_R_32F;
        let b_type = cudaDataType::CUDA_R_32F;
        let c_type = cudaDataType::CUDA_R_32F;
        let alpha = &alpha as *const f32;
        let beta = &beta as *const f32;

        // TODO use impl Gemm<f32> for CudaBlas with DevicePtr<T>.
        let status = unsafe {
            cublasSgemmEx(
                handle, transa, transb, m, n, k, alpha, a, a_type, lda, b, b_type, ldb, beta, c,
                c_type, ldc,
            )
        };
        assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    // TODO return Result
    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        let handle = self.handle;
        let alpha = &alpha as *const f32;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        let status = unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) };
        assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    // TODO return Result
    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
        let handle = self.handle;
        let x = x.as_ptr();
        let y = y.as_ptr();
        let mut result: f32 = 0.0;
        let status = unsafe {
            let result = &mut result as *mut f32;
            cublasSdot_v2(handle, n, x, incx, y, incy, result)
        };
        assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        result
    }

    // TODO return Result
    fn scopy(&self, n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        let handle = self.handle;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        let status = unsafe { cublasScopy_v2(handle, n, x, incx, y, incy) };
        assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    // TODO return Result
    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32) {
        let handle = self.handle;
        let x = x.as_mut_ptr();
        let alpha = &alpha as *const f32;
        let status = unsafe { cublasSscal_v2(handle, n, alpha, x, incx) };
        assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
}
