use std::ops::Deref;
use std::{borrow::BorrowMut, mem::swap};

use rustacuda::memory::CopyDestination;
use rustacuda::memory::DeviceBuffer;

use crate::Error;
use crate::{Device, ErrorEnum};

#[derive(Debug)]
pub struct DevBuffer {
    device: Device,
    buffer: DevBufferEnum,
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
    CudaBuffer(DeviceBuffer<f32>),
}

impl DevBuffer {
    pub fn new(device: &Device, len: usize) -> DevBuffer {
        let buffer = device.device.deref().device_buffer(len);
        DevBuffer {
            device: device.clone(),
            buffer,
        }
    }

    pub fn as_ptr(&self) -> *const f32 {
        match &self.buffer {
            DevBufferEnum::CpuBuffer(ref values) => values.as_ptr(),
            #[cfg(feature = "cuda")]
            DevBufferEnum::CudaBuffer(ref values) => values.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        match self.buffer.borrow_mut() {
            DevBufferEnum::CpuBuffer(ref mut values) => values.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            DevBufferEnum::CudaBuffer(ref mut values) => values.as_mut_ptr(),
        }
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        match &self.buffer {
            DevBufferEnum::CpuBuffer(ref values) => Ok(values.clone()),
            #[cfg(feature = "cuda")]
            DevBufferEnum::CudaBuffer(ref buffer) => {
                let mut values = vec![0.0; buffer.len()];
                match buffer.copy_to(values.as_mut_slice()) {
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
                buffer.copy_from(new_values.as_slice()).unwrap();
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
                    let mut new_buffer = unsafe { DeviceBuffer::uninitialized(new_len).unwrap() };
                    swap(buffer, &mut new_buffer);
                }
            }
        }
    }
}
