use crate::{
    clip::Clip,
    identity::Identity,
    reduce_l2::ReduceL2,
    statistics::bernoulli::Bernoulli,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Add, ClipNorm, Concat, Div, ExecutableOperator, Gemm, Mul, OperatorAttributes, ReduceSum,
    ReduceSumSquare, Reshape, ScalarAdd, ScalarMul, Sigmoid, Softmax, SoftmaxCrossEntropyLoss,
    Sqrt, Sub, Unconcat,
};

#[derive(Clone, Debug)]
pub enum OpCode {
    /// https://onnx.ai/onnx/operators/onnx__Gemm.html
    Gemm,

    /// https://onnx.ai/onnx/operators/onnx__Identity.html
    Identity,

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

    /// https://onnx.ai/onnx/operators/onnx__ReduceL2.html
    ReduceL2,

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
    Reshape,

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
            OpCode::Gemm => "Gemm".to_owned(),
            OpCode::Identity => "Identity".into(),
            OpCode::ReduceSum => "ReduceSum".into(),
            OpCode::Add => "Add".into(),
            OpCode::Sub => "Sub".into(),
            OpCode::Mul => "Mul".into(),
            OpCode::ScalarMul => "ScalarMul".into(),
            OpCode::ScalarAdd => "ScalarAdd".into(),
            OpCode::Clip => "Clip".into(),
            OpCode::ClipNorm => "ClipNorm".into(),
            OpCode::ReduceL2 => "ReduceL2".into(),
            OpCode::Softmax => "Softmax".into(),
            OpCode::Sigmoid => "Sigmoid".into(),
            OpCode::Reshape => "Reshape".into(),
            OpCode::Concat => "Concat".into(),
            OpCode::Unconcat => "Unconcat".into(),
            OpCode::SoftmaxCrossEntropyLoss => "SoftmaxCrossEntropyLoss".into(),
            OpCode::ReduceSumSquare => "ReduceSumSquare".into(),
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
        attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        match self {
            OpCode::Gemm => Gemm::execute(attributes, inputs, outputs, device_stream),
            OpCode::Identity => Identity::execute(attributes, inputs, outputs, device_stream),
            OpCode::ReduceSum => ReduceSum::execute(attributes, inputs, outputs, device_stream),
            OpCode::Add => Add::execute(attributes, inputs, outputs, device_stream),
            OpCode::ScalarMul => ScalarMul::execute(attributes, inputs, outputs, device_stream),
            OpCode::Clip => Clip::execute(attributes, inputs, outputs, device_stream),
            OpCode::ClipNorm => ClipNorm::execute(attributes, inputs, outputs, device_stream),
            OpCode::ReduceL2 => ReduceL2::execute(attributes, inputs, outputs, device_stream),
            OpCode::Mul => Mul::execute(attributes, inputs, outputs, device_stream),
            OpCode::Softmax => Softmax::execute(attributes, inputs, outputs, device_stream),
            OpCode::Sub => Sub::execute(attributes, inputs, outputs, device_stream),
            OpCode::Reshape => Reshape::execute(attributes, inputs, outputs, device_stream),
            OpCode::Concat => Concat::execute(attributes, inputs, outputs, device_stream),
            OpCode::Unconcat => Unconcat::execute(attributes, inputs, outputs, device_stream),
            OpCode::Sigmoid => Sigmoid::execute(attributes, inputs, outputs, device_stream),
            OpCode::SoftmaxCrossEntropyLoss => {
                SoftmaxCrossEntropyLoss::execute(attributes, inputs, outputs, device_stream)
            }
            OpCode::ReduceSumSquare => {
                ReduceSumSquare::execute(attributes, inputs, outputs, device_stream)
            }
            OpCode::Bernoulli => Bernoulli::execute(attributes, inputs, outputs, device_stream),
            OpCode::Div => Div::execute(attributes, inputs, outputs, device_stream),
            OpCode::Sqrt => Sqrt::execute(attributes, inputs, outputs, device_stream),
            OpCode::ScalarAdd => ScalarAdd::execute(attributes, inputs, outputs, device_stream),
        }
    }
}
