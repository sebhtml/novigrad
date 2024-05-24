use crate::{
    Add, ClipNorm, Concat, CrossEntropyLoss, Dropout, Error, Gemm, Mul, ReduceSum, Reshape,
    ResidualSumOfSquares, ScalarMul, Sigmoid, Softmax, Sub, Tensor, Unconcat,
};

#[derive(Clone, Debug)]
pub enum OpCode {
    /// https://onnx.ai/onnx/operators/onnx__Gemm.html
    GemmNTN,
    GemmNNN,
    GemmTNN,
    GemmTNT,

    /// https://onnx.ai/onnx/operators/onnx__ReduceSum.html
    ReduceSum,

    /// https://onnx.ai/onnx/operators/onnx__Add.html
    Add,

    /// Not ONNX-compliant
    /// TODO remove this op code and use Mul with broadcast
    ScalarMul,

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

    /// https://onnx.ai/onnx/operators/onnx__Dropout.html
    Dropout(f32),

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

impl Into<String> for OpCode {
    fn into(self) -> String {
        match self {
            OpCode::GemmNTN => "GemmNTN".into(),
            OpCode::GemmNNN => "GemmNNN".into(),
            OpCode::GemmTNN => "GemmTNN".into(),
            OpCode::GemmTNT => "GemmTNT".into(),
            OpCode::ReduceSum => "ReduceSum".into(),
            OpCode::Add => "Add".into(),
            OpCode::Sub => "Sub".into(),
            OpCode::Mul => "Mul".into(),
            OpCode::ScalarMul => "ScalarMul".into(),
            OpCode::ClipNorm(clipped_norm) => format!("Clip {}", clipped_norm),
            OpCode::Softmax => "Softmax".into(),
            OpCode::Sigmoid => "Sigmoid".into(),
            OpCode::Reshape(output_size) => format!("Reshape {:?}", output_size),
            OpCode::Concat => "Concat".into(),
            OpCode::Unconcat => "Unconcat".into(),
            OpCode::CrossEntropyLoss => "CrossEntropyLoss".into(),
            OpCode::ResidualSumOfSquares => "ResidualSumOfSquares".into(),
            OpCode::Dropout(_) => "Dropout".into(),
        }
        .into()
    }
}

impl OpCode {
    pub fn execute(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        match self {
            OpCode::GemmNTN => Gemm::execute(false, true, false, inputs, outputs),
            OpCode::GemmNNN => Gemm::execute(false, false, false, inputs, outputs),
            OpCode::GemmTNN => Gemm::execute(true, false, false, inputs, outputs),
            OpCode::GemmTNT => Gemm::execute(true, false, true, inputs, outputs),
            OpCode::ReduceSum => ReduceSum::execute(inputs, outputs),
            OpCode::Add => Add::execute(inputs, outputs),
            OpCode::ScalarMul => ScalarMul::execute(inputs, outputs),
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
            OpCode::Dropout(dropout_probability) => {
                Dropout::execute(*dropout_probability, inputs, outputs)
            }
        }
    }
}
