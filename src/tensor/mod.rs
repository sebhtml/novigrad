use std::fmt::Debug;
mod tensor;
#[cfg(feature = "cuda")]
use cudarc::nvrtc::CompileError;
pub use tensor::*;

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

#[macro_export]
macro_rules! error {
    ( $error:expr ) => {
        Error::new(file!(), line!(), column!(), $error)
    };
}

#[derive(Clone, Debug, PartialEq)]
pub enum ErrorEnum {
    IncompatibleTensorShapes,
    UnsupportedOperation,
    IncorrectOperatorConfiguration,
    InputOutputError,
    #[cfg(feature = "cuda")]
    NvRtcCompilePtxError(CompileError),
    #[cfg(feature = "cuda")]
    NvRtcLoadPtxError,
    #[cfg(feature = "cuda")]
    NvGetFuncError(String),
    #[cfg(feature = "cuda")]
    NvLaunchError,
}
