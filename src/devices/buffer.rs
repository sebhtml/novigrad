use std::{borrow::BorrowMut, mem::swap};

use rustacuda::memory::CopyDestination;
use rustacuda::memory::DeviceBuffer;

use crate::Error;

#[derive(Debug)]
pub enum DevBuffer {
    CpuBuffer(Vec<f32>),
    #[cfg(feature = "cuda")]
    CudaBuffer(DeviceBuffer<f32>),
}

impl DevBuffer {
    pub fn as_ptr(&self) -> *const f32 {
        match &self {
            DevBuffer::CpuBuffer(ref values) => values.as_ptr(),
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref values) => values.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        match self.borrow_mut() {
            DevBuffer::CpuBuffer(ref mut values) => values.as_mut_ptr(),
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref mut values) => values.as_mut_ptr(),
        }
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        match &self {
            DevBuffer::CpuBuffer(ref values) => Ok(values.clone()),
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref buffer) => {
                let mut values = vec![0.0; buffer.len()];
                match buffer.copy_to(values.as_mut_slice()) {
                    Ok(_) => Ok(values),
                    _ => Err(Error::UnsupportedOperation),
                }
            }
        }
    }

    pub fn set_values(&mut self, new_values: Vec<f32>) {
        match self.borrow_mut() {
            DevBuffer::CpuBuffer(ref mut values) => {
                values.clear();
                values.extend_from_slice(new_values.as_slice())
            }
            #[cfg(feature = "cuda")]
            DevBuffer::CudaBuffer(ref mut buffer) => {
                // TODO don't unwrap directly.
                buffer.copy_from(new_values.as_slice()).unwrap();
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DevBuffer::CpuBuffer(buffer) => buffer.len(),
            DevBuffer::CudaBuffer(buffer) => buffer.len(),
        }
    }

    pub fn resize(&mut self, new_len: usize) {
        match self {
            DevBuffer::CpuBuffer(buffer) => buffer.resize(new_len, Default::default()),
            DevBuffer::CudaBuffer(buffer) => {
                if buffer.len() != new_len {
                    let mut new_buffer = unsafe { DeviceBuffer::uninitialized(new_len).unwrap() };
                    swap(buffer, &mut new_buffer);
                }
            }
        }
    }
}
