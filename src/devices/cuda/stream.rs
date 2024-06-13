use cudarc::{
    cublas::CudaBlas,
    driver::{CudaSlice, CudaStream},
};

pub struct CudaDeviceStream {
    pub stream: CudaStream,
    pub rng_state: CudaSlice<u64>,
    pub cuda_blas: CudaBlas,
}
