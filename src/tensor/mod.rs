mod dot_product;
use std::fmt::Debug;

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
