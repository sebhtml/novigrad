use super::cpu::slice::CpuDevSlice;
#[cfg(feature = "cuda")]
use super::cuda::slice::CudaDevSlice;
use crate::tensor::Error;
use crate::Device;
use crate::DeviceTrait;
use std::borrow::BorrowMut;

#[derive(Debug)]
pub struct DevSlice {
    pub buffer: DeviceSlice,
}

#[derive(Debug)]
pub enum DeviceSlice {
    CpuDevSlice(CpuDevSlice),
    #[cfg(feature = "cuda")]
    CudaDevSlice(CudaDevSlice),
}

pub trait DevSliceTrait {
    fn as_ptr(&self) -> *const f32;
    fn as_mut_ptr(&mut self) -> *mut f32;
    fn get_values(&self) -> Result<Vec<f32>, Error>;
    fn set_values(&mut self, new_values: Vec<f32>) -> Result<(), Error>;
    fn len(&self) -> usize;
}

impl DevSlice {
    pub fn new(device: &Device, len: usize) -> DevSlice {
        // TODO remove unwrap
        let slice = device.slice(len as i32).unwrap();
        DevSlice { buffer: slice }
    }
}

impl DevSliceTrait for DevSlice {
    fn as_ptr(&self) -> *const f32 {
        match &self.buffer {
            DeviceSlice::CpuDevSlice(ref slice) => slice.as_ptr(),
            #[cfg(feature = "cuda")]
            DeviceSlice::CudaDevSlice(ref slice) => slice.as_ptr(),
        }
    }

    fn as_mut_ptr(&mut self) -> *mut f32 {
        match self.buffer.borrow_mut() {
            DeviceSlice::CpuDevSlice(ref mut slice) => slice.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            DeviceSlice::CudaDevSlice(ref mut slice) => slice.as_mut_ptr(),
        }
    }

    fn get_values(&self) -> Result<Vec<f32>, Error> {
        match self.buffer {
            DeviceSlice::CpuDevSlice(ref slice) => slice.get_values(),
            #[cfg(feature = "cuda")]
            DeviceSlice::CudaDevSlice(ref slice) => slice.get_values(),
        }
    }

    fn set_values(&mut self, new_values: Vec<f32>) -> Result<(), Error> {
        match self.buffer.borrow_mut() {
            DeviceSlice::CpuDevSlice(ref mut slice) => slice.set_values(new_values),
            #[cfg(feature = "cuda")]
            DeviceSlice::CudaDevSlice(ref mut slice) => slice.set_values(new_values),
        }
    }

    fn len(&self) -> usize {
        match &self.buffer {
            DeviceSlice::CpuDevSlice(slice) => slice.len(),
            #[cfg(feature = "cuda")]
            DeviceSlice::CudaDevSlice(slice) => slice.len(),
        }
    }
}
