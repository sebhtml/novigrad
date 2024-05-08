use crate::Gemm;

/// Linear is not a ONNX operator. https://onnx.ai/onnx/operators/index.html ???
pub type Linear = Gemm;
