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
