use crate::{devices::Device, Add, Error, MatMul, OperatorTrait, Tensor};

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

impl OperatorTrait for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 3);
        let input = &inputs[0];
        let weights = &inputs[1];
        let biases = &inputs[2];
        let product = self.matmul.forward(&[input.clone(), weights.clone()])?;
        let sum = self.add.forward(&[product, biases.clone()])?;
        Ok(sum)
    }

    fn forward_realize(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        panic!()
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        panic!()
    }
}
