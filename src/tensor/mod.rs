use std::fmt::Debug;
mod generic_tensor;
use cudarc::nvrtc::CompileError;
pub use generic_tensor::*;

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
    IncorrectOperatorConfiguration,
    InputOutputError,
    NvRtcCompilePtxError(CompileError),
    NvRtcLoadPtxError,
}
