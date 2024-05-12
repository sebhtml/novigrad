use std::{ops::Deref, rc::Rc};

use super::LossFunction;
use crate::{
    devices::Device, BinaryOperator, Error, ErrorEnum, Instruction, Operator, Tensor, TensorF32,
};

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
    fn evaluate(_device: &Device, expected: &TensorF32, actual: &TensorF32) -> Result<f32, Error> {
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
        expected: &TensorF32,
        actual: &TensorF32,
        result: &mut TensorF32,
    ) -> Result<(), Error> {
        TensorF32::copy(actual, result)?;
        TensorF32::sub(expected, result)
    }
}

impl BinaryOperator for CrossEntropyLoss {
    fn forward(&self, input_1: &Tensor, input_2: &Tensor) -> Result<Tensor, Error> {
        let output = self.device.tensor(1, 1, vec![0.0], true, false);
        let inputs = &[input_1, input_2];
        let outputs = &[&output];
        output.push_forward_instruction(Rc::new(self.clone()), inputs, outputs);
        Ok(output)
    }
}

impl Operator for CrossEntropyLoss {
    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let expected = &inputs[0].tensor().deref().borrow();
        let actual = &inputs[1].tensor().deref().borrow();
        let loss = CrossEntropyLoss::evaluate(&self.device, expected, actual)?;
        outputs[0]
            .tensor()
            .deref()
            .borrow()
            .set_values(vec![loss; 1]);
        Ok(())
    }

    fn backward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let instruction = Instruction::new(
            Rc::new(CrossEntropyLossBackward::default()),
            outputs,
            inputs,
        );
        instruction.forward()
    }
}

pub struct CrossEntropyLossBackward {}

impl Default for CrossEntropyLossBackward {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for CrossEntropyLossBackward {
    fn name(&self) -> &str {
        "CrossEntropyLossBackward"
    }

    fn forward(&self, _inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        debug_assert_eq!(outputs.len(), 2);
        if outputs[1].requires_grad() {
            let output_gradient: &mut TensorF32 = &mut outputs[1].gradient().deref().borrow_mut();
            let expected: &TensorF32 = &outputs[0].tensor().deref().borrow();
            let actual: &TensorF32 = &outputs[1].tensor().deref().borrow();
            CrossEntropyLoss::derive(expected, actual, output_gradient)?;
        }
        Ok(())
    }

    fn backward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        todo!()
    }
}
