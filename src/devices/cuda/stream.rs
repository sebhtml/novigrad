use cudarc::driver::CudaStream;

pub struct CudaDeviceStream {
    pub stream: CudaStream,
}
