use crate::{slice::DevSliceTrait, Error};


#[derive(Debug)]
pub struct CpuDevSlice {
    slice: Vec<f32>,
}

impl CpuDevSlice {
    pub fn new(slice: Vec<f32>,) -> Self {
        Self {
            slice,
        }
    }
}

impl DevSliceTrait for CpuDevSlice {
    fn as_ptr(&self) -> *const f32 {
        self.slice.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.slice.as_mut_ptr()
    }

    fn get_values(&self) -> Result<Vec<f32>, Error> {
        Ok(self.slice.clone())
    }

    fn set_values(&mut self, new_values: Vec<f32>) -> Result<(), Error> {
        self.slice.clear();
        self.slice.extend_from_slice(new_values.as_slice());
        Ok(())
    }

    fn len(&self) -> usize {
        self.slice.len()
    }
}