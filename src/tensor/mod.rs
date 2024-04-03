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

#[derive(Clone, Debug, PartialEq)]
pub enum Tensor {
    TensorF32(TensorF32),
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
        Tensor::TensorF32(TensorF32::new_with_int(rows, cols, value))
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
            Tensor::TensorF32(that) => that.into(),
        }
    }
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

impl TensorTrait for Tensor {
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

    fn reshape(&mut self, new_rows: usize, new_cols: usize) {
        match self {
            Tensor::TensorF32(that) => that.reshape(new_rows, new_cols),
        }
    }

    fn values<'a>(&'a self) -> &'a Vec<f32> {
        match self {
            Tensor::TensorF32(that) => that.values(),
        }
    }

    fn int_values<'a>(&'a self) -> &'a Vec<usize> {
        match self {
            Tensor::TensorF32(that) => that.int_values(),
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
}
