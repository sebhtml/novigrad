use std::fmt::Debug;
mod tensor_f32;
pub use tensor_f32::*;

#[cfg(test)]
mod tests;

/*
pub struct Error {
    file: &'static str,
    line: u32,
    column: u32,
    error: ErrorEnum,
}
 */

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    IncompatibleTensorShapes,
    UnsupportedOperation,
}
