use std::ops::Deref;

use super::LossFunction;
use crate::{devices::Device, DeltaWorkingMemory, Error, LearningTensor, OperatorTrait, Tensor};

#[derive(Clone)]
pub struct CrossEntropyLoss {}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self {}
    }
}

const EPSILON: f32 = 1e-8;

impl LossFunction for CrossEntropyLoss {
    /// H(P, Q) = - Σ (P(i) * log(Q(i)))
    fn evaluate(&self, _device: &Device, expected: &Tensor, actual: &Tensor) -> Result<f32, Error> {
        debug_assert_eq!(actual.shape(), expected.shape());
        let p = expected;
        let q = actual;
        if p.shape() != q.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let rows = p.rows();
        debug_assert_eq!(rows, 1);
        let cols = p.cols();
        let mut col = 0;
        let mut sum = 0.0;
        let p_values = p.get_values();
        let q_values = q.get_values();
        while col < cols {
            let p_i = p_values[p.index(0, col)];
            let q_i = q_values[q.index(0, col)] + EPSILON;
            sum += p_i * f32::ln(q_i);
            col += 1;
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
        device: &Device,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        result.assign(device, actual)?;
        Tensor::sub(device, expected, result)
    }
}

impl OperatorTrait for CrossEntropyLoss {
    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<LearningTensor>,
        _output: &LearningTensor,
        _enabled_gradients: &mut Vec<LearningTensor>,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        {
            let expected: &Tensor = &inputs[0].tensor().deref().borrow();
            let actual: &Tensor = &inputs[1].tensor().deref().borrow();
            let backward_gradient: &mut Tensor = &mut inputs[1].gradient().deref().borrow_mut();
            self.derive(device, expected, actual, backward_gradient)?;
        }

        Ok(())
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<LearningTensor>,
    ) -> Result<LearningTensor, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output = device.learning_tensor(0, 0, vec![]);
        let expected: &Tensor = &inputs[0].tensor().deref().borrow();
        let actual: &Tensor = &inputs[1].tensor().deref().borrow();
        let loss = self.evaluate(device, expected, actual)?;
        {
            let output: &mut Tensor = &mut output.tensor().deref().borrow_mut();
            output.reset(1, 1, loss);
        }
        Ok(output)
    }

    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }
}
