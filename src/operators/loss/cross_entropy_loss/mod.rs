use std::{ops::Deref, rc::Rc};

use super::LossFunction;
use crate::{devices::Device, Error, OperatorTrait, Tensor, TensorF32};

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
        debug_assert_eq!(actual.shape(), expected.shape());
        let p = expected;
        let q = actual;
        if p.shape() != q.shape() {
            return Err(Error::IncompatibleTensorShapes);
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
    fn backward(&self, inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let actual: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let backward_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
        self.derive(expected, actual, backward_gradient)?;
        Ok(())
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output = self
            .device
            .tensor(Rc::new(self.clone()), inputs, 1, 1, vec![0.0], false);
        Ok(output)
    }

    fn name(&self) -> &str {
        "CrossEntropyLoss"
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
}
