#[cfg(feature = "cuda")]
use super::cuda::stream::CudaDeviceStream;

pub enum DeviceStream {
    CpuDeviceStream,
    #[cfg(feature = "cuda")]
    CudaDeviceStream(CudaDeviceStream),
}
