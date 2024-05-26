use crate::{
    devices::{Device, DeviceInterface},
    error, DevSliceTrait, Error,
};
use crate::{DevSlice, ErrorEnum};

use std::{cell::RefCell, fmt::Display, ops::Deref, rc::Rc, vec};

#[derive(Clone, Debug)]
pub struct Tensor {
    name: usize,
    device: Device,
    size: Rc<RefCell<Vec<usize>>>,
    device_slice: Rc<RefCell<DevSlice>>,
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
    ) -> Result<Self, Error> {
        debug_assert_eq!(values.len(), rows * cols);
        let mut buffer = device.buffer(values.len());
        buffer.set_values(values)?;
        let tensor = Self {
            name,
            device: device.clone(),
            size: Rc::new(RefCell::new(vec![rows, cols])),
            device_slice: Rc::new(RefCell::new(buffer)),
        };
        Ok(tensor)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn name(&self) -> usize {
        self.name
    }

    pub fn requires_grad(&self) -> bool {
        self.len() > 0
    }

    pub fn rows(&self) -> usize {
        if self.size.deref().borrow().len() == 2 {
            self.size.deref().borrow()[0]
        } else {
            panic!()
        }
    }

    pub fn cols(&self) -> usize {
        if self.size.deref().borrow().len() == 2 {
            self.size.deref().borrow()[1]
        } else {
            panic!()
        }
    }

    pub fn len(&self) -> usize {
        self.size
            .deref()
            .borrow()
            .iter()
            .fold(1, |acc, item| acc * item)
    }

    pub fn size(&self) -> &Rc<RefCell<Vec<usize>>> {
        &self.size
    }

    pub fn index(&self, row: usize, col: usize) -> usize {
        if self.size.deref().borrow().len() == 2 {
            let cols = self.size.deref().borrow()[1];
            row * cols + col
        } else {
            panic!()
        }
    }

    pub fn transpose(&self, other: &mut Tensor) -> Result<(), Error> {
        let self_values = self.get_values()?;
        let mut other_values = other.get_values()?;
        let rows = self.rows();
        let cols = self.cols();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let value = self_values[self.index(row, col)];
                other_values[other.index(col, row)] = value;
                col += 1;
            }
            row += 1;
        }
        other.set_values(other_values)
    }

    pub fn device_slice(&self) -> &Rc<RefCell<DevSlice>> {
        &self.device_slice
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.device_slice.deref().borrow().as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.device_slice.deref().borrow_mut().as_mut_ptr()
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        self.device_slice.deref().borrow().get_values()
    }

    pub fn set_values(&self, new_values: Vec<f32>) -> Result<(), Error> {
        debug_assert_eq!(new_values.len(), self.len());
        if self.device_slice.deref().borrow().len() != self.len() {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        }
        self.device_slice
            .deref()
            .borrow_mut()
            .set_values(new_values)
    }

    pub fn mul(left: &Tensor, right: &Tensor, result: &Tensor) -> Result<(), Error> {
        if left.size() != right.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        if left.size() != result.size() {
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        let device = left.device.clone();
        device.mul(left, right, result)
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

    pub fn copy(x: &Tensor, y: &Tensor) -> Result<(), Error> {
        let device = &x.device;
        let n = x.len() as i32;
        let incx = 1;
        let incy = 1;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        device.copy(n, x, incx, y, incy)
    }

    pub fn copy_slice(
        n: usize,
        src: &Tensor,
        src_row: usize,
        src_col: usize,
        dst: &Tensor,
        dst_row: usize,
        dst_col: usize,
    ) -> Result<(), Error> {
        let device = &src.device;
        let n = n as i32;
        let incx = 1;
        let incy = 1;
        let x = src.as_ptr().wrapping_add(src_row * src.cols() + src_col);
        let y = dst
            .as_mut_ptr()
            .wrapping_add(dst_row * dst.cols() + dst_col);
        device.copy(n, x, incx, y, incy)
    }

    pub fn sub(x: &Tensor, y: &Tensor) -> Result<(), Error> {
        let alpha = -1.0;
        Self::a_x_plus_y(alpha, x, y)
    }

    pub fn add(x: &Tensor, y: &Tensor) -> Result<(), Error> {
        let alpha = 1.0;
        Self::a_x_plus_y(alpha, x, y)
    }

    fn a_x_plus_y(alpha: f32, x: &Tensor, y: &Tensor) -> Result<(), Error> {
        let device = &x.device;
        if x.len() != y.len() {
            println!("Incompatible sizes");
            println!("x {}", x);
            println!("y {}", y);
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        let n = x.len() as i32;
        let incx = 1;
        let incy = 1;
        device.axpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }

    pub fn l2_norm(&self, output: &Tensor) -> Result<(), Error> {
        let device = self.device();
        device.dot(self, self, output)?;
        let squared_l2_norm = output.get_values()?[0];
        let l2_norm = squared_l2_norm.sqrt();
        output.set_values(vec![l2_norm])?;
        Ok(())
    }

    pub fn normalize(&self) -> Result<(), Error> {
        let l2_norm = self.device.tensor(1, 1, vec![0.0]).unwrap();
        self.l2_norm(&l2_norm)?;
        let l2_norm = l2_norm.get_values()?[0];
        if l2_norm == 0.0 {
            return Ok(());
        }
        let alpha = 1.0 / l2_norm;
        let x = self;
        let device = self.device();
        let alpha = device.tensor(1, 1, vec![alpha]).unwrap();
        device.scalar_mul(&alpha, x)
    }

    pub fn resize(&self, new_size: &[usize]) -> Result<(), Error> {
        let new_len = new_size.iter().fold(1, |acc, value| acc * value);
        if new_len != self.len() {
            return Err(error!(ErrorEnum::UnsupportedOperation));
        }

        *self.size.deref().borrow_mut() = new_size.to_owned();

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
            self.size.deref().borrow()
        );
        _ = write!(f, "\n");
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let value = self_values[self.index(row, col)];
                if value < 0.0 {
                    _ = write!(f, " {:2.8}", value);
                } else {
                    _ = write!(f, " +{:2.8}", value);
                }
            }
            _ = write!(f, "\n");
        }
        Ok(())
    }
}

impl TryInto<f32> for &Tensor {
    type Error = Error;

    fn try_into(self) -> Result<f32, Self::Error> {
        let size: &[usize] = &self.size().deref().borrow();
        match size {
            &[1, 1] => {
                let self_values = self.get_values()?;
                Ok(self_values[self.index(0, 0)])
            }
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        }
    }
}
