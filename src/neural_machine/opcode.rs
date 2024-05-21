use crate::{
    Add, ClipNorm, Concat, CrossEntropyLoss, Error, Gemm, Mul, Reshape, ResidualSumOfSquares,
    ScalarMul, Sigmoid, Softmax, Sub, TensorF32, Unconcat,
};

#[derive(Clone, Debug)]
pub enum OpCode {
    /// https://onnx.ai/onnx/operators/onnx__Gemm.html
    Gemm(bool, bool, bool),

    /// https://onnx.ai/onnx/operators/onnx__Add.html
    Add,

    /// Not ONNX-compliant
    /// TODO remove this op code and use Mul with broadcast
    ScalarMul(f32),

    /// Not ONNX-compliant
    /// similar op codes:
    /// - https://onnx.ai/onnx/operators/onnx__Clip.html
    /// - https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
    ClipNorm(f32),

    /// https://onnx.ai/onnx/operators/onnx__Mul.html
    Mul,

    /// https://onnx.ai/onnx/operators/onnx__Softmax.html
    Softmax,

    /// https://onnx.ai/onnx/operators/onnx__Sub.html
    Sub,

    /// https://onnx.ai/onnx/operators/onnx__Reshape.html
    Reshape(Vec<usize>),

    /// https://onnx.ai/onnx/operators/onnx__Sigmoid.html
    Sigmoid,

    /// https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html
    CrossEntropyLoss,

    /// Not ONNX-compliant
    ResidualSumOfSquares,

    /// TODO
    /// https://onnx.ai/onnx/operators/onnx__Dropout.html
    /// Dropout,

    /// TODO
    /// https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
    /// LayerNormalization

    /// TODO
    /// https://onnx.ai/onnx/operators/onnx__Gelu.html
    /// Gelu,

    /// TODO
    /// https://onnx.ai/onnx/operators/onnx__Conv.html
    /// Conv,

    /// https://onnx.ai/onnx/operators/onnx__Concat.html
    Concat,

    /// Not ONNX-compliant
    Unconcat,
}

impl OpCode {
    pub fn name(&self) -> &str {
        match self {
            OpCode::Gemm(_, _, _) => "Gemm",
            OpCode::Add => "Add",
            OpCode::Sub => "Sub",
            OpCode::Mul => "Mul",
            OpCode::ScalarMul(_) => "ScalarMul",
            OpCode::ClipNorm(_) => "Clip",
            OpCode::Softmax => "Softmax",
            OpCode::Sigmoid => "Sigmoid",
            OpCode::Reshape(_) => "Reshape",
            OpCode::Concat => "Concat",
            OpCode::Unconcat => "Unconcat",
            OpCode::CrossEntropyLoss => "CrossEntropyLoss",
            OpCode::ResidualSumOfSquares => "ResidualSumOfSquares",
        }
    }

    pub fn execute(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        match self {
            OpCode::Gemm(trans_a, trans_b, trans_result) => {
                Gemm::execute(*trans_a, *trans_b, *trans_result, inputs, outputs)
            }
            OpCode::Add => Add::execute(inputs, outputs),
            OpCode::ScalarMul(alpha) => ScalarMul::execute(*alpha, inputs, outputs),
            OpCode::ClipNorm(clipped_norm) => ClipNorm::execute(*clipped_norm, inputs, outputs),
            OpCode::Mul => Mul::execute(inputs, outputs),
            OpCode::Softmax => Softmax::execute(inputs, outputs),
            OpCode::Sub => Sub::execute(inputs, outputs),
            OpCode::Reshape(output_size) => Reshape::execute(output_size, inputs, outputs),
            OpCode::Concat => Concat::execute(inputs, outputs),
            OpCode::Unconcat => Unconcat::execute(inputs, outputs),
            OpCode::Sigmoid => Sigmoid::execute(inputs, outputs),
            OpCode::CrossEntropyLoss => CrossEntropyLoss::execute(inputs, outputs),
            OpCode::ResidualSumOfSquares => ResidualSumOfSquares::execute(inputs, outputs),
        }
    }
}
