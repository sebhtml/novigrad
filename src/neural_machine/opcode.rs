use crate::{
    clip::Clip, statistics::bernoulli::Bernoulli, Add, ClipNorm, Concat, Div, Error, Gemm, Mul,
    ReduceSum, ReduceSumSquare, Reshape, ScalarAdd, ScalarMul, Sigmoid, Softmax,
    SoftmaxCrossEntropyLoss, Sqrt, Sub, Tensor, Unconcat,
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
    /// TODO remove this op code and use Add with broadcast
    ScalarAdd,

    /// Not ONNX-compliant
    /// TODO remove this op code and use Mul with broadcast
    ScalarMul,

    /// https://onnx.ai/onnx/operators/onnx__Clip.html
    Clip,

    /// https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
    /// LayerNormalization

    /// Not ONNX-compliant
    /// https://onnx.ai/onnx/operators/onnx__Clip.html
    ClipNorm,

    /// https://onnx.ai/onnx/operators/onnx__Mul.html
    Mul,

    /// https://onnx.ai/onnx/operators/onnx__Div.html
    Div,

    /// https://onnx.ai/onnx/operators/onnx__Sqrt.html
    Sqrt,

    /// https://onnx.ai/onnx/operators/onnx__Softmax.html
    Softmax,

    /// https://onnx.ai/onnx/operators/onnx__Sub.html
    Sub,

    /// https://onnx.ai/onnx/operators/onnx__Reshape.html
    Reshape(Vec<usize>),

    /// https://onnx.ai/onnx/operators/onnx__Sigmoid.html
    Sigmoid,

    /// https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html
    SoftmaxCrossEntropyLoss,

    /// https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html
    ReduceSumSquare,

    /// https://onnx.ai/onnx/operators/onnx__Bernoulli.html
    Bernoulli,

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
            OpCode::ScalarAdd => "ScalarAdd".into(),
            OpCode::Clip => "Clip".into(),
            OpCode::ClipNorm => "ClipNorm".into(),
            OpCode::Softmax => "Softmax".into(),
            OpCode::Sigmoid => "Sigmoid".into(),
            OpCode::Reshape(output_size) => format!("Reshape {:?}", output_size),
            OpCode::Concat => "Concat".into(),
            OpCode::Unconcat => "Unconcat".into(),
            OpCode::SoftmaxCrossEntropyLoss => "CrossEntropyLoss".into(),
            OpCode::ReduceSumSquare => "ResidualSumOfSquares".into(),
            OpCode::Bernoulli => "Bernoulli".into(),
            OpCode::Div => "Div".into(),
            OpCode::Sqrt => "Sqrt".into(),
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
            OpCode::Clip => Clip::execute(inputs, outputs),
            OpCode::ClipNorm => ClipNorm::execute(inputs, outputs),
            OpCode::Mul => Mul::execute(inputs, outputs),
            OpCode::Softmax => Softmax::execute(inputs, outputs),
            OpCode::Sub => Sub::execute(inputs, outputs),
            OpCode::Reshape(output_size) => Reshape::execute(output_size, inputs, outputs),
            OpCode::Concat => Concat::execute(inputs, outputs),
            OpCode::Unconcat => Unconcat::execute(inputs, outputs),
            OpCode::Sigmoid => Sigmoid::execute(inputs, outputs),
            OpCode::SoftmaxCrossEntropyLoss => SoftmaxCrossEntropyLoss::execute(inputs, outputs),
            OpCode::ReduceSumSquare => ReduceSumSquare::execute(inputs, outputs),
            OpCode::Bernoulli => Bernoulli::execute(inputs, outputs),
            OpCode::Div => Div::execute(inputs, outputs),
            OpCode::Sqrt => Sqrt::execute(inputs, outputs),
            OpCode::ScalarAdd => ScalarAdd::execute(inputs, outputs),
        }
    }
}
