use std::rc::Rc;

use crate::{
    Add, ClipNorm, Concat, ConcatBackward, CrossEntropyLoss, Error, Gemm, Mul, Operator, Reshape,
    ScalarMul, ScalarMulBackward, Sigmoid, Softmax, Sub, TensorF32,
};

#[derive(Clone, Debug)]
pub enum OpCode {
    // Not ONNX-compliant
    // TODO remove this op code
    DynOperator(Rc<dyn Operator>),

    /// https://onnx.ai/onnx/operators/onnx__Gemm.html
    Gemm(bool, bool, bool),

    /// https://onnx.ai/onnx/operators/onnx__Add.html
    Add,

    /// Not ONNX-compliant
    /// TODO remove this op code and use Mul with broadcast
    ScalarMul(f32),

    /// Not ONNX-compliant
    /// TODO remove this op code
    ScalarMulBackward,

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

    /// TODO
    /// Not ONNX-compliant
    /// TODO remove this op code
    /// CrossEntropyLossBackward, // TODO

    /// TODO
    /// Not ONNX-compliant
    /// TODO remove this op code
    // ResidualSumOfSquares, // TODO

    /// TODO
    /// Not ONNX-compliant
    /// TODO remove this op code
    // ResidualSumOfSquaresBackward, // TODO

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
    ConcatBackward,
}

impl Operator for OpCode {
    fn name(&self) -> &str {
        match self {
            OpCode::DynOperator(inner) => inner.name(),
            OpCode::Gemm(_, _, _) => "Gemm",
            OpCode::Add => "Add",
            OpCode::ScalarMul(_) => "ScalarMul",
            OpCode::ScalarMulBackward => "ScalarMulBackward",
            OpCode::ClipNorm(_) => "Clip",
            OpCode::Mul => "Mul",
            OpCode::Softmax => "Softmax",
            OpCode::Sub => "Sub",
            OpCode::Reshape(_) => "Reshape",
            OpCode::Concat => "Concat",
            OpCode::ConcatBackward => "ConcatBackward",
            OpCode::Sigmoid => "Sigmoid",
            OpCode::CrossEntropyLoss => "CrossEntropyLoss",
        }
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        match self {
            OpCode::DynOperator(inner) => inner.forward(inputs, outputs),
            OpCode::Gemm(trans_a, trans_b, trans_result) => {
                Gemm::execute(*trans_a, *trans_b, *trans_result, inputs, outputs)
            }
            OpCode::Add => Add::execute(inputs, outputs),
            OpCode::ScalarMul(alpha) => ScalarMul::execute(*alpha, inputs, outputs),
            OpCode::ScalarMulBackward => ScalarMulBackward::execute(inputs, outputs),
            OpCode::ClipNorm(clipped_norm) => ClipNorm::execute(*clipped_norm, inputs, outputs),
            OpCode::Mul => Mul::execute(inputs, outputs),
            OpCode::Softmax => Softmax::execute(inputs, outputs),
            OpCode::Sub => Sub::execute(inputs, outputs),
            OpCode::Reshape(output_size) => Reshape::execute(output_size, inputs, outputs),
            OpCode::Concat => Concat::execute(inputs, outputs),
            OpCode::ConcatBackward => ConcatBackward::execute(inputs, outputs),
            OpCode::Sigmoid => Sigmoid::execute(inputs, outputs),
            OpCode::CrossEntropyLoss => CrossEntropyLoss::execute(inputs, outputs),
        }
    }
}
