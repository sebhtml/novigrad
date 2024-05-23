use std::{borrow::BorrowMut, mem::swap};

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice};

use crate::error;
use crate::DeviceInterface;
use crate::Error;
use crate::{Device, ErrorEnum};

#[derive(Debug)]
pub struct DevSlice {
    device: Device,
    pub buffer: DevSliceEnum,
}

impl Drop for DevSlice {
    fn drop(&mut self) {
        if self.len() == 0 {
            return;
        }
        let device = self.device.clone();
        device.recycle(self.len(), self);
    }
}

#[derive(Debug)]
pub enum DevSliceEnum {
    CpuDevSlice(Vec<f32>),
    #[cfg(feature = "cuda")]
    CudaDevSlice(CudaSlice<f32>),
}

impl DevSlice {
    pub fn new(device: &Device, len: usize) -> DevSlice {
        // TODO remove unwrap
        let slice = device.slice(len as i32).unwrap();
        DevSlice {
            device: device.clone(),
            buffer: slice,
        }
    }

    pub fn as_ptr(&self) -> *const f32 {
        match &self.buffer {
            DevSliceEnum::CpuDevSlice(ref values) => values.as_ptr(),
            #[cfg(feature = "cuda")]
            DevSliceEnum::CudaDevSlice(ref values) => *values.device_ptr() as *const _,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        match self.buffer.borrow_mut() {
            DevSliceEnum::CpuDevSlice(ref mut values) => values.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            DevSliceEnum::CudaDevSlice(ref mut values) => *values.device_ptr_mut() as *mut _,
        }
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        match self.buffer {
            DevSliceEnum::CpuDevSlice(ref values) => Ok(values.clone()),
            #[cfg(feature = "cuda")]
            DevSliceEnum::CudaDevSlice(ref buffer) => {
                let mut values = vec![0.0; buffer.len()];
                let dev = buffer.device();
                let result = dev.dtoh_sync_copy_into(buffer, &mut values);
                match result {
                    Ok(_) => Ok(values),
                    _ => Err(error!(ErrorEnum::UnsupportedOperation)),
                }
            }
        }
    }

    pub fn set_values(&mut self, new_values: Vec<f32>) -> Result<(), Error> {
        match self.buffer.borrow_mut() {
            DevSliceEnum::CpuDevSlice(ref mut values) => {
                values.clear();
                values.extend_from_slice(new_values.as_slice());
                Ok(())
            }
            #[cfg(feature = "cuda")]
            DevSliceEnum::CudaDevSlice(ref mut buffer) => {
                let dev = buffer.device();
                dev.htod_sync_copy_into(&new_values, buffer)
                    .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
            }
        }
    }

    pub fn len(&self) -> usize {
        match &self.buffer {
            DevSliceEnum::CpuDevSlice(buffer) => buffer.len(),
            DevSliceEnum::CudaDevSlice(buffer) => buffer.len(),
        }
    }

    pub fn resize(&mut self, new_len: usize) {
        match self.buffer.borrow_mut() {
            DevSliceEnum::CpuDevSlice(buffer) => buffer.resize(new_len, Default::default()),
            DevSliceEnum::CudaDevSlice(buffer) => {
                if buffer.len() != new_len {
                    let dev = buffer.device();
                    // TODO don't unwrap
                    let mut new_buffer = dev.alloc_zeros(new_len).unwrap();
                    swap(buffer, &mut new_buffer);
                }
            }
        }
    }
}
