use crate::{
    devices::{Device, DeviceInterface},
    Error,
};
use crate::{DevBuffer, ErrorEnum};

use std::fmt::Display;

#[derive(Debug)]
pub struct TensorF32 {
    device: Device,
    size: Vec<usize>,
    buffer: DevBuffer,
}

impl PartialEq for TensorF32 {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows()
            && self.cols() == other.cols()
            && self.get_values() == other.get_values()
    }
}

impl TensorF32 {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>, device: &Device) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut buffer = device.buffer(values.len());
        buffer.set_values(values);
        Self {
            device: device.clone(),
            size: vec![rows, cols],
            buffer,
        }
    }

    pub fn rows(&self) -> usize {
        if self.size.len() == 2 {
            self.size[0]
        } else {
            panic!()
        }
    }

    pub fn cols(&self) -> usize {
        if self.size.len() == 2 {
            self.size[1]
        } else {
            panic!()
        }
    }

    pub fn len(&self) -> usize {
        self.size.iter().fold(1, |acc, item| acc * item)
    }

    pub fn size(&self) -> &[usize] {
        &self.size
    }

    pub fn index(&self, row: usize, col: usize) -> usize {
        if self.size.len() == 2 {
            let cols = self.size[1];
            row * cols + col
        } else {
            panic!()
        }
    }

    pub fn transpose(&self, other: &mut TensorF32) -> Result<(), Error> {
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
        other.set_values(other_values);
        Ok(())
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.buffer.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.buffer.as_mut_ptr()
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        self.buffer.get_values()
    }

    pub fn reallocate(&mut self, new_size: &[usize]) {
        if new_size == self.size {
            return;
        }
        self.size = new_size.to_owned();
        let len = self.len();
        self.set_values(vec![0.0; len]);
    }

    pub fn set_values(&mut self, new_values: Vec<f32>) {
        debug_assert_eq!(new_values.len(), self.len());
        if self.buffer.len() != self.len() {
            self.buffer.resize(self.len())
        }
        self.buffer.set_values(new_values)
    }

    pub fn zero(&mut self) -> Result<(), Error> {
        TensorF32::scale(0.0, self)
    }

    // TODO use device for element_wise_mul
    pub fn mul(left: &TensorF32, right: &TensorF32, result: &mut TensorF32) -> Result<(), Error> {
        if left.size() != right.size() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }

        debug_assert_eq!(result.size(), left.size());
        result.zero()?;

        let mut result_values = result.get_values()?;
        let left_values = left.get_values()?;
        let right_values = right.get_values()?;

        let result_ptr = result_values.as_mut_ptr();
        let left_ptr = left_values.as_ptr();
        let right_ptr = right_values.as_ptr();

        unsafe {
            let mut index = 0;
            let len = left_values.len();
            while index < len {
                let left_cell = left_ptr.add(index);
                let right_cell = right_ptr.add(index);
                let result_cell = result_ptr.add(index);
                let left = *left_cell;
                let right = *right_cell;
                let value = left * right;
                //debug_assert!(value.is_finite());
                *result_cell = value;
                index += 1;
            }
        }

        result.set_values(result_values);

        Ok(())
    }

    pub fn dot_product(x: &TensorF32, y: &TensorF32) -> Result<f32, Error> {
        let device = &x.device;
        if x.size() != y.size() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }
        let n = x.len() as i32;
        let incx = 1;
        let incy = 1;
        device.sdot(n, x.as_ptr(), incx, y.as_ptr(), incy)
    }

    pub fn copy(x: &TensorF32, y: &mut TensorF32) -> Result<(), Error> {
        let device = &x.device;
        let n = x.len() as i32;
        let incx = 1;
        let incy = 1;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        device.scopy(n, x, incx, y, incy)
    }

    pub fn matmul(
        transa: bool,
        transb: bool,
        a: &TensorF32,
        b: &TensorF32,
        c: &mut TensorF32,
        transpose_result: bool,
    ) -> Result<(), Error> {
        c.zero()?;
        let alpha = 1.0;
        let beta = 0.0;
        TensorF32::gemm(transa, transb, alpha, a, b, beta, c, transpose_result)
    }

    pub fn gemm(
        transa: bool,
        transb: bool,
        alpha: f32,
        a: &TensorF32,
        b: &TensorF32,
        beta: f32,
        c: &mut TensorF32,
        transpose_result: bool,
    ) -> Result<(), Error> {
        let op_result = Self::_gemm(transa, transb, alpha, a, b, beta, c, transpose_result);
        match op_result {
            Ok(value) => Ok(value),
            Err(error) => {
                let device = &a.device;
                let mut b_t = device.tensor_f32(b.cols(), b.rows(), vec![0.0; b.cols() * b.rows()]);
                b.transpose(&mut b_t)?;
                println!("Incompatible shapes in matrix multiplication");
                println!("transa: {}, transb: {}", transa, transb);
                println!("Between A {:?} and B^T {:?}", a.size(), b_t.size(),);
                debug_assert!(false);
                Err(error)
            }
        }
    }

    fn _gemm(
        transa: bool,
        transb: bool,
        alpha: f32,
        a: &TensorF32,
        b: &TensorF32,
        beta: f32,
        c: &mut TensorF32,
        transpose_result: bool,
    ) -> Result<(), Error> {
        let device = &a.device;
        if !transa && !transb && !transpose_result {
            if a.cols() != b.rows() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            let (m, n, k) = (a.rows(), b.cols(), a.cols());
            device.sgemm(
                false,
                false,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b.as_ptr(),
                n as i32,
                a.as_ptr(),
                k as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            )
        } else if transa && !transb && !transpose_result {
            if a.rows() != b.rows() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            let (m, n, k) = (a.cols(), b.cols(), a.rows());

            device.sgemm(
                false,
                true,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b.as_ptr(),
                n as i32,
                a.as_ptr(),
                a.cols() as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            )
        } else if !transa && transb && !transpose_result {
            if a.cols() != b.cols() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            let (m, n, k) = (a.rows(), b.rows(), a.cols());

            device.sgemm(
                true,
                false,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b.as_ptr(),
                b.cols() as i32,
                a.as_ptr(),
                k as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            )
        } else if transa && transb && !transpose_result {
            if a.rows() != b.cols() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            let (m, n, k) = (a.cols(), b.rows(), a.rows());

            device.sgemm(
                true,
                true,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b.as_ptr(),
                b.cols() as i32,
                a.as_ptr(),
                a.cols() as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            )
        } else if transa && transb && transpose_result {
            if a.rows() != b.cols() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            let (m, n, k) = (a.cols(), b.rows(), a.rows());

            device.sgemm(
                false,
                false,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                a.cols() as i32,
                b.as_ptr(),
                b.cols() as i32,
                beta,
                c.as_mut_ptr(),
                m as i32,
            )
        } else if transa && !transb && transpose_result {
            if a.rows() != b.rows() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            let (m, n, k) = (a.cols(), b.cols(), a.rows());

            device.sgemm(
                false,
                true,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                a.cols() as i32,
                b.as_ptr(),
                b.cols() as i32,
                beta,
                c.as_mut_ptr(),
                m as i32,
            )
        } else {
            Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::UnsupportedOperation,
            ))
        }
    }

    pub fn sub(x: &TensorF32, y: &mut TensorF32) -> Result<(), Error> {
        let alpha = -1.0;
        Self::a_x_plus_y(alpha, x, y)
    }

    pub fn add(x: &TensorF32, y: &mut TensorF32) -> Result<(), Error> {
        let alpha = 1.0;
        Self::a_x_plus_y(alpha, x, y)
    }

    pub fn a_x_plus_y(alpha: f32, x: &TensorF32, y: &mut TensorF32) -> Result<(), Error> {
        let device = &x.device;
        if x.len() != y.len() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }
        let n = x.len() as i32;
        let incx = 1;
        let incy = 1;
        device.saxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy)
    }

    // TODO use device to clip
    pub fn clip(&self, min: f32, max: f32, result: &mut TensorF32) -> Result<(), Error> {
        let len = self.len();
        let mut index = 0;
        let self_values = self.get_values()?;
        let mut result_values = result.get_values()?;
        while index < len {
            let mut value = self_values[index];
            value = value.max(min);
            value = value.min(max);
            result_values[index] = value;
            index += 1;
        }
        result.set_values(result_values);
        Ok(())
    }

    pub fn scale(alpha: f32, x: &mut TensorF32) -> Result<(), Error> {
        let device = x.device.clone();
        let n = x.len() as i32;
        let incx = 1;
        device.sscal(n, alpha, x.as_mut_ptr(), incx)
    }

    pub fn resize(&mut self, new_size: &[usize]) -> Result<(), Error> {
        let new_len = new_size.iter().fold(1, |acc, value| acc * value);
        if new_len != self.len() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::UnsupportedOperation,
            ));
        }

        self.size = new_size.to_owned();

        Ok(())
    }
}

impl Display for TensorF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let self_values = self.get_values().map_err(|_| std::fmt::Error)?;
        _ = write!(f, "Shape: {:?}", self.size);
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

impl TryInto<f32> for &TensorF32 {
    type Error = Error;

    fn try_into(self) -> Result<f32, Self::Error> {
        match self.size() {
            &[1, 1] => {
                let self_values = self.get_values()?;
                Ok(self_values[self.index(0, 0)])
            }
            _ => Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::UnsupportedOperation,
            )),
        }
    }
}
