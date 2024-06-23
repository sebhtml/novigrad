use crate::{
    new_tensor,
    tensor::{Error, Tensor},
    Device,
};

#[cfg(feature = "cuda")]
use super::cuda::stream::CudaDeviceStream;

pub struct DeviceStream {
    pub variant: DeviceStreamEnum,
    // Working memory of the stream
    // TODO design a better way to do this as the number of temporary tensors increases.
    pub max_alpha: Tensor,
    pub l2_norm: Tensor,
    pub one: Tensor,
    pub alpha: Tensor,
    pub zero: Tensor,
}

impl DeviceStream {
    pub fn try_new(device: &Device, variant: DeviceStreamEnum) -> Result<Self, Error> {
        let that = Self {
            variant,
            max_alpha: new_tensor!(device, 1, 1, vec![1.0],)?,
            l2_norm: new_tensor!(device, 1, 1, vec![0.0],)?,
            one: new_tensor!(device, 1, 1, vec![1.0],)?,
            alpha: new_tensor!(device, 1, 1, vec![0.0],)?,
            zero: new_tensor!(device, 1, 1, vec![0.0],)?,
        };
        Ok(that)
    }
}

pub enum DeviceStreamEnum {
    CpuDeviceStream,
    #[cfg(feature = "cuda")]
    CudaDeviceStream(CudaDeviceStream),
}

pub trait StreamTrait {
    fn wait_for_default(&self) -> Result<(), Error>;
    fn wait_for(&self) -> Result<(), Error>;
}

impl StreamTrait for DeviceStream {
    fn wait_for_default(&self) -> Result<(), Error> {
        match &self.variant {
            DeviceStreamEnum::CpuDeviceStream => Ok(()),
            #[cfg(feature = "cuda")]
            DeviceStreamEnum::CudaDeviceStream(stream) => stream.wait_for_default(),
        }
    }
    fn wait_for(&self) -> Result<(), Error> {
        match &self.variant {
            DeviceStreamEnum::CpuDeviceStream => Ok(()),
            #[cfg(feature = "cuda")]
            DeviceStreamEnum::CudaDeviceStream(stream) => stream.wait_for(),
        }
    }
}
