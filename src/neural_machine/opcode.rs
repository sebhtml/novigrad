use crate::{
    clip::Clip,
    identity::Identity,
    statistics::bernoulli::Bernoulli,
    tensor::{Error, Tensor},
    Add, ClipNorm, Concat, Div, Gemm, Mul, ReduceSum, ReduceSumSquare, Reshape, ScalarAdd,
    ScalarMul, Sigmoid, Softmax, SoftmaxCrossEntropyLoss, Sqrt, Sub, Unconcat,
};

#[derive(Clone, Debug)]
pub enum OpCode {
    /// https://onnx.ai/onnx/operators/onnx__Gemm.html
    Gemm(bool, bool, bool),

    /// https://onnx.ai/onnx/operators/onnx__Identity.html
    Identity(String),

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
            OpCode::Gemm(_, _, _) => "Gemm".into(),
            OpCode::Identity(label) => "Identity".to_string() + " label=" + label.as_str(),
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
    pub fn execute(
        &self,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: usize,
    ) -> Result<(), Error> {
        match self {
            OpCode::Gemm(trans_a, trans_b, trans_result) => Gemm::execute(
                *trans_a,
                *trans_b,
                *trans_result,
                inputs,
                outputs,
                device_stream,
            ),
            OpCode::Identity(_) => Identity::execute(inputs, outputs, device_stream),
            OpCode::ReduceSum => ReduceSum::execute(inputs, outputs, device_stream),
            OpCode::Add => Add::execute(inputs, outputs, device_stream),
            OpCode::ScalarMul => ScalarMul::execute(inputs, outputs, device_stream),
            OpCode::Clip => Clip::execute(inputs, outputs, device_stream),
            OpCode::ClipNorm => ClipNorm::execute(inputs, outputs, device_stream),
            OpCode::Mul => Mul::execute(inputs, outputs, device_stream),
            OpCode::Softmax => Softmax::execute(inputs, outputs, device_stream),
            OpCode::Sub => Sub::execute(inputs, outputs, device_stream),
            OpCode::Reshape(output_size) => {
                Reshape::execute(output_size, inputs, outputs, device_stream)
            }
            OpCode::Concat => Concat::execute(inputs, outputs, device_stream),
            OpCode::Unconcat => Unconcat::execute(inputs, outputs, device_stream),
            OpCode::Sigmoid => Sigmoid::execute(inputs, outputs, device_stream),
            OpCode::SoftmaxCrossEntropyLoss => {
                SoftmaxCrossEntropyLoss::execute(inputs, outputs, device_stream)
            }
            OpCode::ReduceSumSquare => ReduceSumSquare::execute(inputs, outputs, device_stream),
            OpCode::Bernoulli => Bernoulli::execute(inputs, outputs, device_stream),
            OpCode::Div => Div::execute(inputs, outputs, device_stream),
            OpCode::Sqrt => Sqrt::execute(inputs, outputs, device_stream),
            OpCode::ScalarAdd => ScalarAdd::execute(inputs, outputs, device_stream),
        }
    }
}
