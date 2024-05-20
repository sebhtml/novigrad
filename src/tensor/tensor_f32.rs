use crate::{
    devices::{Device, DeviceInterface},
    Error,
};
use crate::{DevBuffer, ErrorEnum};

use std::{cell::RefCell, fmt::Display, ops::Deref, rc::Rc, vec};

#[derive(Clone, Debug)]
pub struct TensorF32 {
    name: usize,
    device: Device,
    size: Rc<RefCell<Vec<usize>>>,
    buffer: Rc<RefCell<DevBuffer>>,
}

impl PartialEq for TensorF32 {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows()
            && self.cols() == other.cols()
            && self.get_values() == other.get_values()
    }
}

impl TensorF32 {
    pub fn new(name: usize, rows: usize, cols: usize, values: Vec<f32>, device: &Device) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut buffer = device.buffer(values.len());
        buffer.set_values(values);
        Self {
            name,
            device: device.clone(),
            size: Rc::new(RefCell::new(vec![rows, cols])),
            buffer: Rc::new(RefCell::new(buffer)),
        }
    }

    pub fn name(&self) -> String {
        "t".to_owned() + self.name.to_string().as_str()
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
        self.buffer.deref().borrow().as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.buffer.deref().borrow_mut().as_mut_ptr()
    }

    pub fn get_values(&self) -> Result<Vec<f32>, Error> {
        self.buffer.deref().borrow().get_values()
    }

    pub fn reallocate(&mut self, new_size: &[usize]) {
        if *new_size == *self.size.deref().borrow() {
            return;
        }
        *self.size.deref().borrow_mut() = new_size.to_owned();
        let len = self.len();
        self.set_values(vec![0.0; len]);
    }

    // TODO This method should return a Result.
    pub fn set_values(&self, new_values: Vec<f32>) {
        debug_assert_eq!(new_values.len(), self.len());
        if self.buffer.deref().borrow().len() != self.len() {
            self.buffer.deref().borrow_mut().resize(self.len())
        }
        self.buffer.deref().borrow_mut().set_values(new_values)
    }

    pub fn zero(&self) -> Result<(), Error> {
        TensorF32::scale(0.0, self)
    }

    // TODO use device for element_wise_mul
    pub fn mul(left: &TensorF32, right: &TensorF32, result: &TensorF32) -> Result<(), Error> {
        if left.size() != right.size() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }

        debug_assert_eq!(result.size(), left.size());

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
                *result_cell = value;
                index += 1;
            }
        }

        result.set_values(result_values);

        Ok(())
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

    pub fn copy(x: &TensorF32, y: &TensorF32) -> Result<(), Error> {
        let device = &x.device;
        let n = x.len() as i32;
        let incx = 1;
        let incy = 1;
        let x = x.as_ptr();
        let y = y.as_mut_ptr();
        device.scopy(n, x, incx, y, incy)
    }

    pub fn copy_slice(
        src: &TensorF32,
        src_row: usize,
        src_col: usize,
        dst: &TensorF32,
        dst_row: usize,
        dst_col: usize,
    ) -> Result<(), Error> {
        let device = &src.device;
        let n = src.cols() as i32;
        let incx = 1;
        let incy = 1;
        let x = src.as_ptr().wrapping_add(src_row * src.cols() + src_col);
        let y = dst
            .as_mut_ptr()
            .wrapping_add(dst_row * dst.cols() + dst_col);
        device.scopy(n, x, incx, y, incy)
    }

    pub fn gemm(
        transa: bool,
        transb: bool,
        alpha: f32,
        a: &TensorF32,
        b: &TensorF32,
        beta: f32,
        c: &TensorF32,
        transpose_result: bool,
    ) -> Result<(), Error> {
        let op_result = Self::_gemm(transa, transb, alpha, a, b, beta, c, transpose_result);
        match op_result {
            Ok(value) => Ok(value),
            Err(error) => {
                println!("Incompatible sizes in GEMM");
                println!(
                    "transa: {}, transb: {}, transpose_result: {}",
                    transa, transb, transpose_result
                );
                println!(
                    "A size: {:?}  B size:  {:?}  C size:  {:?}",
                    a.size().deref().borrow(),
                    b.size().deref().borrow(),
                    c.size().deref().borrow(),
                );
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
        c: &TensorF32,
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
            if a.rows() != c.rows() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            if b.cols() != c.cols() {
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
            if a.cols() != c.rows() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            if b.cols() != c.cols() {
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
            if a.rows() != c.rows() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            if b.rows() != c.cols() {
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
            if a.cols() != c.rows() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            if b.rows() != c.cols() {
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
            if a.cols() != c.cols() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            if b.rows() != c.rows() {
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
            if a.cols() != c.cols() {
                return Err(Error::new(
                    file!(),
                    line!(),
                    column!(),
                    ErrorEnum::IncompatibleTensorShapes,
                ));
            }
            if b.cols() != c.rows() {
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

    pub fn sub(x: &TensorF32, y: &TensorF32) -> Result<(), Error> {
        let alpha = -1.0;
        Self::a_x_plus_y(alpha, x, y)
    }

    pub fn add(x: &TensorF32, y: &TensorF32) -> Result<(), Error> {
        let alpha = 1.0;
        Self::a_x_plus_y(alpha, x, y)
    }

    fn a_x_plus_y(alpha: f32, x: &TensorF32, y: &TensorF32) -> Result<(), Error> {
        let device = &x.device;
        if x.len() != y.len() {
            println!("Incompatible sizes");
            println!("x {}", x);
            println!("y {}", y);
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

    pub fn l2_norm(&self) -> Result<f32, Error> {
        let squared_l2_norm = TensorF32::dot_product(self, self)?;
        let l2_norm = squared_l2_norm.sqrt();
        Ok(l2_norm)
    }

    pub fn clip(&self, norm: f32) -> Result<(), Error> {
        let l2_norm = self.l2_norm()?;
        if l2_norm == 0.0 {
            return Ok(());
        }
        let alpha = 1.0 / l2_norm * norm;
        let x = self;
        TensorF32::scale(alpha, x)
    }

    pub fn scale(alpha: f32, x: &TensorF32) -> Result<(), Error> {
        let device = x.device.clone();
        let n = x.len() as i32;
        let incx = 1;
        device.sscal(n, alpha, x.as_mut_ptr(), incx)
    }

    pub fn resize(&self, new_size: &[usize]) -> Result<(), Error> {
        let new_len = new_size.iter().fold(1, |acc, value| acc * value);
        if new_len != self.len() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::UnsupportedOperation,
            ));
        }

        *self.size.deref().borrow_mut() = new_size.to_owned();

        Ok(())
    }
}

impl Display for TensorF32 {
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

impl TryInto<f32> for &TensorF32 {
    type Error = Error;

    fn try_into(self) -> Result<f32, Self::Error> {
        let size: &[usize] = &self.size().deref().borrow();
        match size {
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
