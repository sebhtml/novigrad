use std::{borrow::BorrowMut, mem::swap};

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice};

use crate::DeviceInterface;
use crate::Error;
use crate::{Device, ErrorEnum};

#[derive(Debug)]
pub struct DevBuffer {
    device: Device,
    pub buffer: DevBufferEnum,
}

impl Drop for DevBuffer {
    fn drop(&mut self) {
        if self.len() == 0 {
            return;
        }
        let device = self.device.clone();
        device.recycle(self.len(), self);
    }
}

#[derive(Debug)]
pub enum DevBufferEnum {
    CpuBuffer(Vec<f32>),
    #[cfg(feature = "cuda")]
    CudaBuffer(CudaSlice<f32>),
}

impl DevBuffer {
    pub fn new(device: &Device, len: usize) -> DevBuffer {
        // TODO remove unwrap
        let slice = device.slice(len as i32).unwrap();
        DevBuffer {
            device: device.clone(),
            buffer: slice,
        }
    }

    pub fn as_ptr(&self) -> *const f32 {
        match &self.buffer {
            DevBufferEnum::CpuBuffer(ref values) => values.as_ptr(),
            #[cfg(feature = "cuda")]
            DevBufferEnum::CudaBuffer(ref values) => *values.device_ptr() as *const _,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        match self.buffer.borrow_mut() {
            DevBufferEnum::CpuBuffer(ref mut values) => values.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            DevBufferEnum::CudaBuffer(ref mut values) => *values.device_ptr_mut() as *mut _,
        }
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        match self.buffer {
            DevBufferEnum::CpuBuffer(ref values) => Ok(values.clone()),
            #[cfg(feature = "cuda")]
            DevBufferEnum::CudaBuffer(ref buffer) => {
                let mut values = vec![0.0; buffer.len()];
                let dev = buffer.device();
                let result = dev.dtoh_sync_copy_into(buffer, &mut values);
                match result {
                    Ok(_) => Ok(values),
                    _ => Err(Error::new(
                        file!(),
                        line!(),
                        column!(),
                        ErrorEnum::UnsupportedOperation,
                    )),
                }
            }
        }
    }

    pub fn set_values(&mut self, new_values: Vec<f32>) {
        match self.buffer.borrow_mut() {
            DevBufferEnum::CpuBuffer(ref mut values) => {
                values.clear();
                values.extend_from_slice(new_values.as_slice())
            }
            #[cfg(feature = "cuda")]
            DevBufferEnum::CudaBuffer(ref mut buffer) => {
                // TODO don't unwrap directly.
                let dev = buffer.device();
                dev.htod_sync_copy_into(&new_values, buffer).unwrap();
            }
        }
    }

    pub fn len(&self) -> usize {
        match &self.buffer {
            DevBufferEnum::CpuBuffer(buffer) => buffer.len(),
            DevBufferEnum::CudaBuffer(buffer) => buffer.len(),
        }
    }

    pub fn resize(&mut self, new_len: usize) {
        match self.buffer.borrow_mut() {
            DevBufferEnum::CpuBuffer(buffer) => buffer.resize(new_len, Default::default()),
            DevBufferEnum::CudaBuffer(buffer) => {
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
