use crate::{
    new_tensor_with_grad, tensor::Error, Add, BinaryOperator, Device, Mul, TensorWithGrad,
    UnaryOperator,
};

use super::standardization::Standardization;

/// See
/// LayerNormalization
/// https://arxiv.org/pdf/1607.06450
/// https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
pub struct LayerNormalization {
    standardization: Standardization,
    mul: Mul,
    add: Add,
    gain: TensorWithGrad,
    bias: TensorWithGrad,
}

impl LayerNormalization {
    pub fn try_new(device: &Device, rows: usize, cols: usize) -> Result<Self, Error> {
        let gain =
            new_tensor_with_grad!(device, rows, cols, vec![1.0; rows * cols], &[], true, true)?;
        let bias =
            new_tensor_with_grad!(device, rows, cols, vec![0.0; rows * cols], &[], true, true)?;
        let standardization = Standardization::new(device);
        let mul = Mul::new(device);
        let add = Add::new(device);
        let op = Self {
            standardization,
            mul,
            add,
            gain,
            bias,
        };
        Ok(op)
    }
}

impl UnaryOperator for LayerNormalization {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, crate::tensor::Error> {
        let standardized = self.standardization.forward(input)?;
        let with_gain = self.mul.forward(&self.gain, &standardized)?;
        let with_bias = self.add.forward(&with_gain, &self.bias)?;
        Ok(with_bias)
    }
}
