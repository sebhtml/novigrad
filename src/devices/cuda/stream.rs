use cudarc::driver::{CudaSlice, CudaStream};

pub struct CudaDeviceStream {
    pub stream: CudaStream,
    pub rng_state: CudaSlice<u64>,
}
