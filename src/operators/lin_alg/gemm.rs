use crate::{devices::Device, Error, Operator, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Gemm.html
#[derive(Clone)]
pub struct Gemm {
    transb: bool,
}

impl Gemm {
    pub fn new(_device: &Device, transb: bool) -> Self {
        Self { transb }
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
        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);
        let input = inputs[0];
        let weights = inputs[1];
        let biases = outputs[0];
        let a = input;
        let b = weights;
        let c = biases;
        let transb = self.transb;
        TensorF32::gemm(false, transb, 1.0, a, b, 1.0, c, false)
    }
}
