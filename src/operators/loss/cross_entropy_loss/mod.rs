use std::{ops::Deref, rc::Rc};

use super::LossFunction;
use crate::{devices::Device, Error, ErrorEnum, OperatorTrait, Tensor, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html
#[derive(Clone)]
pub struct CrossEntropyLoss {
    device: Device,
}

impl CrossEntropyLoss {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

const EPSILON: f32 = 1e-8;

impl LossFunction for CrossEntropyLoss {
    /// H(P, Q) = - Σ (P(i) * log(Q(i)))
    fn evaluate(
        &self,
        _device: &Device,
        expected: &TensorF32,
        actual: &TensorF32,
    ) -> Result<f32, Error> {
        debug_assert_eq!(actual.size(), expected.size());
        let p = expected;
        let q = actual;
        if p.size() != q.size() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }
        let rows = p.rows();
        let cols = p.cols();
        let mut row = 0;
        let mut col = 0;
        let mut sum = 0.0;
        let p_values = p.get_values()?;
        let q_values = q.get_values()?;
        while row < rows {
            while col < cols {
                let p_i = p_values[p.index(row, col)];
                let q_i = q_values[q.index(row, col)] + EPSILON;
                sum += p_i * f32::ln(q_i);
                col += 1;
            }
            row += 1;
        }
        debug_assert!(sum.is_finite());
        Ok(-sum)
    }

    /// When Cross-Entropy Loss is used with a Softmax activation function,
    /// then we don't need to derive the softmax activations.
    /// The derivative of the Loss in respect to logits (before activation) is
    /// output of the softmax function - expected output (one-hot encoded)
    fn derive(
        &self,
        expected: &TensorF32,
        actual: &TensorF32,
        result: &mut TensorF32,
    ) -> Result<(), Error> {
        TensorF32::copy(actual, result)?;
        TensorF32::sub(expected, result)
    }
}

impl OperatorTrait for CrossEntropyLoss {
    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output =
            self.device
                .tensor(Rc::new(self.clone()), inputs, 1, 1, vec![0.0], true, false);
        Ok(output)
    }

    fn forward_realize(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        let expected: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let actual: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let loss = self.evaluate(&self.device, expected, actual)?;
        output
            .tensor()
            .deref()
            .borrow_mut()
            .set_values(vec![loss; 1]);
        Ok(())
    }

    fn backward(&self, inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        if inputs[1].requires_grad() {
            let input_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
            let expected: &TensorF32 = &inputs[0].tensor().deref().borrow();
            let actual: &TensorF32 = &inputs[1].tensor().deref().borrow();
            self.derive(expected, actual, input_gradient)?;
        }
        Ok(())
    }
}
