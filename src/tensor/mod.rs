use std::fmt::Debug;
mod tensor_f32;
pub use tensor_f32::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    file: &'static str,
    line: u32,
    column: u32,
    error: ErrorEnum,
}

impl Error {
    pub fn new(file: &'static str, line: u32, column: u32, error: ErrorEnum) -> Self {
        Self {
            file,
            line,
            column,
            error,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ErrorEnum {
    IncompatibleTensorShapes,
    UnsupportedOperation,
}
