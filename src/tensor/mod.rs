mod dot_product;
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

pub trait TensorTrait {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row(&self, row: usize, result: &mut Tensor);
    fn index(&self, row: usize, col: usize) -> usize;
    fn shape(&self) -> (usize, usize);
    fn reshape(&mut self, new_rows: usize, new_cols: usize);
    fn values<'a>(&'a self) -> &'a Vec<f32>;
    fn int_values<'a>(&'a self) -> &'a Vec<usize>;
    fn get(&self, row: usize, col: usize) -> f32;
    fn set(&mut self, row: usize, col: usize, value: f32);
    fn assign(&mut self, from: &Tensor);
    fn transpose(&self, other: &mut Tensor);
    fn add(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error>;
    fn sub(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error>;
    fn element_wise_mul(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error>;
    fn div(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error>;
    fn matmul(lhs: &Tensor, rhs: &Tensor, result: &mut Tensor, options: u32) -> Result<(), Error>;
    fn clip(&self, min: f32, max: f32, result: &mut Tensor);
    fn scalar_add(&self, right: f32, result: &mut Tensor) -> Result<(), Error>;
    fn scalar_mul(&self, right: f32, result: &mut Tensor) -> Result<(), Error>;
}
