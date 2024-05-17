use crate::{devices::Device, Error, Operator, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Gemm.html
#[derive(Clone)]
pub struct Gemm {
    trans_a: bool,
    trans_b: bool,
    trans_result: bool,
}

impl Gemm {
    pub fn new(_device: &Device, transa: bool, transb: bool, transpose_result: bool) -> Self {
        Self {
            trans_a: transa,
            trans_b: transb,
            trans_result: transpose_result,
        }
    }

    pub fn execute(
        trans_a: bool,
        trans_b: bool,
        trans_result: bool,
        inputs: &[&crate::TensorF32],
        outputs: &[&crate::TensorF32],
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);
        let input = inputs[0];
        let weights = inputs[1];
        let biases = outputs[0];
        let a = input;
        let b = weights;
        let c = biases;
        let transa = trans_a;
        let transb = trans_b;
        let transpose_result = trans_result;
        TensorF32::gemm(transa, transb, 1.0, a, b, 1.0, c, transpose_result)
    }
}

impl Operator for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn forward(
        &self,
        inputs: &[&crate::TensorF32],
        outputs: &[&crate::TensorF32],
    ) -> Result<(), Error> {
        Self::execute(
            self.trans_a,
            self.trans_b,
            self.trans_result,
            inputs,
            outputs,
        )
    }
}
