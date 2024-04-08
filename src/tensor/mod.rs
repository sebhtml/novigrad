mod dot_product;
use std::fmt::{Debug, Display};

pub use dot_product::*;
mod tensor_f32;
pub use tensor_f32::*;

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
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Self {
        Tensor::TensorF32(TensorF32::new(rows, cols, values))
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Tensor::TensorF32(Default::default())
    }
}

impl<'a> Into<&'a Vec<f32>> for &'a Tensor {
    fn into(self) -> &'a Vec<f32> {
        match self {
            Tensor::TensorF32(that) => that.values(),
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tensor::TensorF32(that) => Display::fmt(&that, f),
        }
    }
}

pub trait TensorTrait<T, Rhs> {
    fn is_finite(&self) -> bool;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row(&self, row: usize, result: &mut Rhs);
    fn col(&self, col: usize, result: &mut Rhs);
    fn index(&self, row: usize, col: usize) -> usize;
    fn shape(&self) -> (usize, usize);
    fn reset(&mut self, new_rows: usize, new_cols: usize, value: T);
    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error>;
    fn values<'a>(&'a self) -> &'a Vec<T>;
    fn get(&self, row: usize, col: usize) -> T;
    fn set(&mut self, row: usize, col: usize, value: T);
    fn assign(&mut self, from: &Rhs);
    fn transpose(&self, other: &mut Rhs);
    fn add(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn add_to_row(&mut self, row: usize, rhs: &Rhs) -> Result<(), Error>;
    fn sub(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn element_wise_mul(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn div(&self, right: &Rhs, result: &mut Rhs) -> Result<(), Error>;
    fn matmul(lhs: &Rhs, rhs: &Rhs, result: &mut Rhs, options: u32) -> Result<(), Error>;
    fn clip(&self, min: T, max: T, result: &mut Rhs);
    fn scalar_add(&self, right: T, result: &mut Rhs) -> Result<(), Error>;
    fn scalar_mul(&self, right: T, result: &mut Rhs) -> Result<(), Error>;
}

impl TensorTrait<f32, Tensor> for Tensor {
    fn is_finite(&self) -> bool {
        match self {
            Tensor::TensorF32(that) => that.is_finite(),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Tensor::TensorF32(that) => that.rows(),
        }
    }

    fn cols(&self) -> usize {
        match self {
            Tensor::TensorF32(that) => that.cols(),
        }
    }

    fn row(&self, row: usize, result: &mut Tensor) {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.row(row, result),
        }
    }

    fn col(&self, col: usize, result: &mut Tensor) {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.col(col, result),
        }
    }

    fn index(&self, row: usize, col: usize) -> usize {
        match self {
            Tensor::TensorF32(that) => that.index(row, col),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Tensor::TensorF32(that) => that.shape(),
        }
    }

    fn reset(&mut self, new_rows: usize, new_cols: usize, value: f32) {
        match self {
            Tensor::TensorF32(that) => that.reset(new_rows, new_cols, value),
        }
    }

    fn values<'a>(&'a self) -> &'a Vec<f32> {
        match self {
            Tensor::TensorF32(that) => that.values(),
        }
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        match self {
            Tensor::TensorF32(that) => that.get(row, col),
        }
    }

    fn set(&mut self, row: usize, col: usize, value: f32) {
        match self {
            Tensor::TensorF32(that) => that.set(row, col, value),
        }
    }

    fn assign(&mut self, from: &Tensor) {
        match (self, from) {
            (Tensor::TensorF32(that), Tensor::TensorF32(from)) => that.assign(from),
        }
    }

    fn transpose(&self, other: &mut Tensor) {
        match (self, other) {
            (Tensor::TensorF32(that), Tensor::TensorF32(other)) => that.transpose(other),
        }
    }

    fn add(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.add(right, result)
            }
        }
    }

    fn sub(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.sub(right, result)
            }
        }
    }

    fn element_wise_mul(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.element_wise_mul(right, result)
            }
        }
    }

    fn div(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        match (self, right, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(right), Tensor::TensorF32(result)) => {
                that.div(right, result)
            }
        }
    }

    fn matmul(lhs: &Tensor, rhs: &Tensor, result: &mut Tensor, options: u32) -> Result<(), Error> {
        match (lhs, rhs, result) {
            (Tensor::TensorF32(lhs), Tensor::TensorF32(rhs), Tensor::TensorF32(result)) => {
                TensorF32::matmul(lhs, rhs, result, options)
            }
        }
    }

    fn clip(&self, min: f32, max: f32, result: &mut Tensor) {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.clip(min, max, result),
        }
    }

    fn scalar_add(&self, right: f32, result: &mut Tensor) -> Result<(), Error> {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.scalar_add(right, result),
        }
    }

    fn scalar_mul(&self, right: f32, result: &mut Tensor) -> Result<(), Error> {
        match (self, result) {
            (Tensor::TensorF32(that), Tensor::TensorF32(result)) => that.scalar_mul(right, result),
        }
    }

    fn add_to_row(&mut self, row: usize, rhs: &Tensor) -> Result<(), Error> {
        match (self, rhs) {
            (Tensor::TensorF32(that), Tensor::TensorF32(rhs)) => that.add_to_row(row, rhs),
        }
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error> {
        match self {
            Tensor::TensorF32(that) => that.reshape(new_rows, new_cols),
        }
    }
}
