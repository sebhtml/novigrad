use crate::{devices::Device, Add, BinaryOperator, Error, MatMul, Tensor, TernaryOperator};

/// https://onnx.ai/onnx/operators/onnx__Gemm.html
#[derive(Clone)]
pub struct Gemm {
    matmul: MatMul,
    add: Add,
}

impl Gemm {
    pub fn new(device: &Device, transb: bool) -> Self {
        if !transb {
            panic!();
        }
        Self {
            matmul: MatMul::new(device, transb),
            add: Add::new(device),
        }
    }
}

impl TernaryOperator for Gemm {
    fn forward(&self, input: &Tensor, weights: &Tensor, biases: &Tensor) -> Result<Tensor, Error> {
        let product = self.matmul.forward(input, weights)?;
        let sum = self.add.forward(&product, biases)?;
        Ok(sum)
    }
}
