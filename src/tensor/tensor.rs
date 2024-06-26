use crate::devices::slice::DevSliceTrait;
use crate::tensor::ErrorEnum;
use crate::{devices::Device, error, slice::DevSlice, tensor::Error};

use std::fmt;
use std::sync::{Arc, RwLock};
use std::{fmt::Display, ops::Deref, vec};

#[derive(Clone)]
pub struct Tensor {
    name: usize,
    size: Arc<RwLock<Vec<usize>>>,
    device_slice: Arc<RwLock<DevSlice>>,
    #[cfg(debug_assertions)]
    file: String,
    #[cfg(debug_assertions)]
    line: u32,
    #[cfg(debug_assertions)]
    column: u32,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Tensor")?;
        writeln!(f, "name: {}", self.name)?;
        writeln!(f, "size: {:?}", self.size.deref().read().unwrap())?;
        writeln!(f, "slice: {:?}", self.get_values())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows()
            && self.cols() == other.cols()
            && self.get_values() == other.get_values()
    }
}

impl Tensor {
    pub fn new(
        name: usize,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        device: &Device,
        #[cfg(debug_assertions)] file: &str,
        #[cfg(debug_assertions)] line: u32,
        #[cfg(debug_assertions)] column: u32,
    ) -> Result<Self, Error> {
        debug_assert_eq!(values.len(), rows * cols);
        let mut buffer = device.buffer(values.len());
        buffer.set_values(values)?;
        let tensor = Self {
            name,
            size: Arc::new(RwLock::new(vec![rows, cols])),
            device_slice: Arc::new(RwLock::new(buffer)),
            #[cfg(debug_assertions)]
            file: file.into(),
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column,
        };
        Ok(tensor)
    }

    pub fn name(&self) -> usize {
        self.name
    }

    pub fn requires_grad(&self) -> bool {
        self.len() > 0
    }

    pub fn rows(&self) -> usize {
        if self.size.deref().read().unwrap().len() == 2 {
            self.size.deref().read().unwrap()[0]
        } else {
            panic!()
        }
    }

    pub fn cols(&self) -> usize {
        if self.size.deref().read().unwrap().len() == 2 {
            self.size.deref().read().unwrap()[1]
        } else {
            panic!()
        }
    }

    pub fn len(&self) -> usize {
        self.size.deref().read().unwrap().iter().product::<usize>()
    }

    pub fn size(&self) -> impl Deref<Target = Vec<usize>> + '_ {
        self.size.read().unwrap()
    }

    pub fn index(&self, row: usize, col: usize) -> usize {
        Self::get_index(&self.size.deref().read().unwrap(), row, col)
    }

    pub fn get_index(size: &[usize], row: usize, col: usize) -> usize {
        if size.len() == 2 {
            let cols = size[1];
            row * cols + col
        } else {
            panic!()
        }
    }

    pub fn device_slice(&self) -> impl Deref<Target = DevSlice> + '_ {
        self.device_slice.read().unwrap()
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.device_slice.deref().read().unwrap().as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.device_slice.deref().write().unwrap().as_mut_ptr()
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        self.device_slice.deref().read().unwrap().get_values()
    }

    /// Avoid using set_values unless necessary. It's bad for performance.
    pub fn set_values(&self, new_values: Vec<f32>) -> Result<(), Error> {
        debug_assert_eq!(new_values.len(), self.len());
        if self.device_slice.deref().read().unwrap().len() != self.len() {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        }
        self.device_slice
            .deref()
            .write()
            .unwrap()
            .set_values(new_values)
    }

    pub fn is_finite(&self) -> bool {
        let values = self.get_values().unwrap();
        for value in values {
            if !value.is_finite() {
                return false;
            }
        }
        true
    }

    pub fn is_nan(&self) -> Result<bool, Error> {
        let values = self.get_values()?;
        for value in values {
            if value.is_nan() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn is_infinite(&self) -> Result<bool, Error> {
        let values = self.get_values()?;
        for value in values {
            if value.is_infinite() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn resize(&self, new_size: &[usize]) -> Result<(), Error> {
        let new_len = new_size.iter().product::<usize>();
        if new_len != self.len() {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        }

        *self.size.deref().write().unwrap() = new_size.to_owned();

        Ok(())
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let self_values = self.get_values().map_err(|_| std::fmt::Error)?;
        _ = write!(
            f,
            "Tensor name: {}, size: {:?}",
            self.name(),
            self.size.deref().read().unwrap()
        );
        _ = writeln!(f);

        #[cfg(debug_assertions)]
        {
            _ = write!(
                f,
                "Tensor file: {}, line: {}, column: {}",
                self.file, self.line, self.column,
            );
            _ = writeln!(f);
        }
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let value = self_values[self.index(row, col)];
                if value < 0.0 {
                    _ = write!(f, " {:2.8}", value);
                } else {
                    _ = write!(f, " +{:2.8}", value);
                }
            }
            _ = writeln!(f);
        }

        Ok(())
    }
}

impl TryInto<f32> for &Tensor {
    type Error = Error;

    fn try_into(self) -> Result<f32, Self::Error> {
        let size: &[usize] = &self.size();
        match size {
            &[1, 1] => {
                let self_values = self.get_values()?;
                Ok(self_values[self.index(0, 0)])
            }
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }
}
