use std::sync::Arc;

use cudarc::{
    cublas::CudaBlas,
    driver::{CudaDevice, CudaSlice, CudaStream},
};

use crate::{
    error,
    stream::StreamTrait,
    tensor::{Error, ErrorEnum},
};

pub struct CudaDeviceStream {
    pub device: Arc<CudaDevice>,
    pub stream: CudaStream,
    pub rng_state: CudaSlice<u64>,
    pub cuda_blas: CudaBlas,
    pub workspace: CudaSlice<u8>,
}

impl StreamTrait for CudaDeviceStream {
    fn synchronize(&self) -> Result<(), crate::tensor::Error> {
        self.device
            .wait_for(&self.stream)
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }
}
