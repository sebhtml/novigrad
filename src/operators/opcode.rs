use crate::{
    analysis::min::Min,
    clip::Clip,
    gelu::{Gelu, GeluDerivative},
    identity::Identity,
    reduce_l2::ReduceL2,
    reduce_sum::ReduceSum,
    statistics::{bernoulli::Bernoulli, standardization::Standardization},
    stream::DeviceStream,
    tensor::{Error, Tensor},
    transpose::Transpose,
    Add, ClipNorm, Concat, Device, Div, ExecutableOperator, Gemm, Mul, OperatorAttributes,
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

    /// Not ONNX-compliant
    /// Equivalent to:
    /// ClipNorm(x)
    ///   norm = ReduceL2(x)
    ///   if norm != 0
    ///     alpha = 1.0 / norm
    ///     x = ScalarMul(alpha, x)
    ///     return x
    ClipNorm,

    /// Not ONNX-compliant
    /// First stage of https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
    Standardization,

    /// https://onnx.ai/onnx/operators/onnx__Transpose.html
    Transpose,

    /// https://onnx.ai/onnx/operators/onnx__ReduceL2.html
    ReduceL2,

    /// https://onnx.ai/onnx/operators/onnx__Mul.html
    Mul,

    /// https://onnx.ai/onnx/operators/onnx__Div.html
    Div,

    /// https://onnx.ai/onnx/operators/onnx__Sqrt.html
    Sqrt,

    /// https://onnx.ai/onnx/operators/onnx__Min.html
    Min,

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

    Gelu,
    GeluDerivative,

    /// TODO
    /// https://onnx.ai/onnx/operators/onnx__Conv.html
    /// Conv,

    /// https://onnx.ai/onnx/operators/onnx__Concat.html
    Concat,

    /// Not ONNX-compliant
    Unconcat,
}

impl From<&OpCode> for String {
    fn from(value: &OpCode) -> String {
        match value {
            OpCode::Gemm => "Gemm".to_owned(),
            OpCode::Identity => "Identity".into(),
            OpCode::ReduceSum => "ReduceSum".into(),
            OpCode::Add => "Add".into(),
            OpCode::Sub => "Sub".into(),
            OpCode::Mul => "Mul".into(),
            OpCode::Div => "Div".into(),
            OpCode::Min => "Min".into(),
            OpCode::ScalarMul => "ScalarMul".into(),
            OpCode::ScalarAdd => "ScalarAdd".into(),
            OpCode::Clip => "Clip".into(),
            OpCode::ClipNorm => "ClipNorm".into(),
            OpCode::ReduceL2 => "ReduceL2".into(),
            OpCode::Standardization => "Standardization".into(),
            OpCode::Softmax => "Softmax".into(),
            OpCode::Sigmoid => "Sigmoid".into(),
            OpCode::Gelu => "Gelu".into(),
            OpCode::GeluDerivative => "GeluDerivative".into(),
            OpCode::Reshape => "Reshape".into(),
            OpCode::Concat => "Concat".into(),
            OpCode::Unconcat => "Unconcat".into(),
            OpCode::SoftmaxCrossEntropyLoss => "SoftmaxCrossEntropyLoss".into(),
            OpCode::ReduceSumSquare => "ReduceSumSquare".into(),
            OpCode::Bernoulli => "Bernoulli".into(),
            OpCode::Sqrt => "Sqrt".into(),
            OpCode::Transpose => "Transpose".into(),
        }
    }
}

impl OpCode {
    pub fn execute(
        &self,
        attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        match self {
            OpCode::Gemm => Gemm::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Identity => {
                Identity::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Add => Add::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Mul => Mul::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Sub => Sub::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Div => Div::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Min => Min::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Reshape => Reshape::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Concat => Concat::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Unconcat => {
                Unconcat::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Sigmoid => Sigmoid::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::Gelu => Gelu::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::GeluDerivative => {
                GeluDerivative::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Standardization => {
                Standardization::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Softmax => Softmax::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::SoftmaxCrossEntropyLoss => {
                SoftmaxCrossEntropyLoss::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::ReduceSumSquare => {
                ReduceSumSquare::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::ReduceL2 => {
                ReduceL2::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::ReduceSum => {
                ReduceSum::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Bernoulli => {
                Bernoulli::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Sqrt => Sqrt::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::ScalarAdd => {
                ScalarAdd::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::ScalarMul => {
                ScalarMul::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Transpose => {
                Transpose::execute(attributes, inputs, outputs, device, device_stream)
            }
            OpCode::Clip => Clip::execute(attributes, inputs, outputs, device, device_stream),
            OpCode::ClipNorm => {
                ClipNorm::execute(attributes, inputs, outputs, device, device_stream)
            }
        }
    }
}
