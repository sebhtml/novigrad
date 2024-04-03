mod dot_product;
use std::{borrow::BorrowMut, fmt::Debug, mem::swap};

pub use dot_product::*;
mod tensor_f32;
pub use tensor_f32::*;
mod tensor_usize;
pub use tensor_usize::*;

#[cfg(test)]
mod tests;

#[derive(Debug, PartialEq)]
pub enum Error {
    IncompatibleTensorShapes,
    UnsupportedOperation,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tensor {
    TensorF32(TensorF32),
    TensorUSize(TensorUSize),
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Self {
        Tensor::TensorF32(TensorF32::new(rows, cols, values))
    }
}

impl From<Vec<usize>> for Tensor {
    fn from(value: Vec<usize>) -> Self {
        let rows = value.len();
        let cols = 1;
        Tensor::TensorUSize(TensorUSize::new(rows, cols, value))
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Tensor::TensorF32(Default::default())
    }
}

impl<'a> Into<&'a Vec<usize>> for &'a Tensor {
    fn into(self) -> &'a Vec<usize> {
        match self {
            Tensor::TensorUSize(that) => that.int_values(),
            _ => panic!("Not implemented"),
        }
    }
}

pub trait TensorTrait<Rhs> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row(&self, row: usize, result: &mut Rhs);
    fn index(&self, row: usize, col: usize) -> usize;
    fn shape(&self) -> (usize, usize);
    fn reshape(&mut self, new_rows: usize, new_cols: usize);
    fn values<'a>(&'a self) -> &'a Vec<f32>;
    fn int_values<'a>(&'a self) -> &'a Vec<usize>;
    fn get(&self, row: usize, col: usize) -> f32;
    fn set(&mut self, row: usize, col: usize, value: f32);
    fn assign(&mut self, from: &Rhs);
    fn transpose(&self, other: &mut Rhs);
    fn add(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn sub(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn element_wise_mul(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn div(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn matmul(lhs: &Rhs, rhs: &Rhs, result: &mut Rhs, options: u32) -> Result<(), Error>;
    fn clip(&self, min: f32, max: f32, result: &mut Rhs);
    fn scalar_add(&self, right: f32, result: &mut Rhs) -> Result<(), Error>;
    fn scalar_mul(&self, right: f32, result: &mut Rhs) -> Result<(), Error>;
}

impl TensorTrait<Tensor> for Tensor {
    fn rows(&self) -> usize {
        match self {
            Tensor::TensorF32(that) => that.rows(),
            _ => panic!("Not implemented"),
        }
    }

    fn cols(&self) -> usize {
        match self {
            Tensor::TensorF32(that) => that.cols(),
            _ => panic!("Not implemented"),
        }
    }

    fn row(&self, row: usize, result: &mut Tensor) {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.row(row, result),
            _ => panic!("Not implemented"),
        }
    }

    fn index(&self, row: usize, col: usize) -> usize {
        match self {
            Tensor::TensorF32(that) => that.index(row, col),
            _ => panic!("Not implemented"),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Tensor::TensorF32(that) => that.shape(),
            _ => panic!("Not implemented"),
        }
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) {
        match self {
            Tensor::TensorF32(that) => that.reshape(new_rows, new_cols),
            _ => panic!("Not implemented"),
        }
    }

    fn values<'a>(&'a self) -> &'a Vec<f32> {
        match self {
            Tensor::TensorF32(that) => that.values(),
            _ => panic!("Not implemented"),
        }
    }

    fn int_values<'a>(&'a self) -> &'a Vec<usize> {
        match self {
            Tensor::TensorUSize(that) => that.int_values(),
            _ => panic!("Not implemented"),
        }
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        match self {
            Tensor::TensorF32(that) => that.get(row, col),
            _ => panic!("Not implemented"),
        }
    }

    fn set(&mut self, row: usize, col: usize, value: f32) {
        match self {
            Tensor::TensorF32(that) => that.set(row, col, value),
            _ => panic!("Not implemented"),
        }
    }

    fn assign(&mut self, from: &Tensor) {
        match (self.borrow_mut(), from) {
            (Tensor::TensorF32(that), Tensor::TensorF32(from)) => that.assign(from),
            (Tensor::TensorUSize(that), Tensor::TensorUSize(from)) => that.assign(from),
            // TODO the type swap outside of here.
            (Tensor::TensorF32(_), Tensor::TensorUSize(_)) => {
                let new_self = Tensor::TensorUSize(Default::default());
                *self = new_self;
                self.assign(from);
            }
            (Tensor::TensorUSize(_), Tensor::TensorF32(_)) => {
                let new_self = Tensor::TensorF32(Default::default());
                *self = new_self;
                self.assign(from);
            }
        }
    }

    fn transpose(&self, other: &mut Tensor) {
        match (self, other) {
            (Tensor::TensorF32(that), Tensor::TensorF32(other)) => that.transpose(other),
            _ => panic!("Not implemented"),
        }
    }

    fn add(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.add(right, result)
            }
            _ => panic!("Not implemented"),
        }
    }

    fn sub(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.sub(right, result)
            }
            _ => panic!("Not implemented"),
        }
    }

    fn element_wise_mul(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.element_wise_mul(right, result)
            }
            _ => panic!("Not implemented"),
        }
    }

    fn div(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.div(right, result)
            }
            _ => panic!("Not implemented"),
        }
    }

    fn matmul(lhs: &Tensor, rhs: &Tensor, result: &mut Tensor, options: u32) -> Result<(), Error> {
        match (lhs, rhs, result) {
            (Tensor::TensorF32(lhs), Tensor::TensorF32(rhs), Tensor::TensorF32(result)) => {
                TensorF32::matmul(lhs, rhs, result, options)
            }
            _ => panic!("Not implemented"),
        }
    }

    fn clip(&self, min: f32, max: f32, result: &mut Tensor) {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.clip(min, max, result),
            _ => panic!("Not implemented"),
        }
    }

    fn scalar_add(&self, right: f32, result: &mut Tensor) -> Result<(), Error> {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.scalar_add(right, result),
            _ => panic!("Not implemented"),
        }
    }

    fn scalar_mul(&self, right: f32, result: &mut Tensor) -> Result<(), Error> {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.scalar_mul(right, result),
            _ => panic!("Not implemented"),
        }
    }
}
