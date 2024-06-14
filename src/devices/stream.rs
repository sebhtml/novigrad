use crate::tensor::Error;

#[cfg(feature = "cuda")]
use super::cuda::stream::CudaDeviceStream;

pub enum DeviceStream {
    CpuDeviceStream,
    #[cfg(feature = "cuda")]
    CudaDeviceStream(CudaDeviceStream),
}

pub trait StreamTrait {
    fn synchronize(&self) -> Result<(), Error>;
}

impl StreamTrait for DeviceStream {
    fn synchronize(&self) -> Result<(), Error> {
        match self {
            DeviceStream::CpuDeviceStream => Ok(()),
            #[cfg(feature = "cuda")]
            DeviceStream::CudaDeviceStream(stream) => stream.synchronize(),
        }
    }
}
