use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice};

use crate::{error, slice::DevSliceTrait, ErrorEnum};

#[derive(Debug)]
pub struct CudaDevSlice {
    slice: CudaSlice<f32>,
}

impl CudaDevSlice {
    pub fn new(slice: CudaSlice<f32>) -> Self {
        Self { slice }
    }
    pub fn slice(&self) -> &CudaSlice<f32> {
        &self.slice
    }
}

impl DevSliceTrait for CudaDevSlice {
    fn as_ptr(&self) -> *const f32 {
        *self.slice.device_ptr() as *const _
    }

    fn as_mut_ptr(&mut self) -> *mut f32 {
        *self.slice.device_ptr_mut() as *mut _
    }

    fn get_values(&self) -> Result<Vec<f32>, crate::Error> {
        let mut values = vec![0.0; self.slice.len()];
        let dev = self.slice.device();
        let result = dev.dtoh_sync_copy_into(&self.slice, &mut values);
        match result {
            Ok(_) => Ok(values),
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }

    fn set_values(&mut self, new_values: Vec<f32>) -> Result<(), crate::Error> {
        let dev = self.slice.device();
        dev.htod_sync_copy_into(&new_values, &mut self.slice)
            .map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn len(&self) -> usize {
        self.slice.len()
    }
}
