use std::{ops::Deref, rc::Rc};

use super::LossFunction;
use crate::{devices::Device, BinaryOperator, Error, ErrorEnum, Operator, Tensor, TensorF32, Zero};

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
            println!("Incompatible sizes");
            println!("p {}", p);
            println!("q {}", q);
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
    fn derive(expected: &TensorF32, actual: &TensorF32, result: &TensorF32) -> Result<(), Error> {
        TensorF32::copy(actual, result)?;
        TensorF32::sub(expected, result)
    }
}

impl BinaryOperator for CrossEntropyLoss {
    fn forward(&self, input_1: &Tensor, input_2: &Tensor) -> Result<Tensor, Error> {
        let output = self
            .device
            .tensor(1, 1, vec![0.0], &[input_1, input_2], true, false);
        let inputs = [input_1, input_2];
        let outputs = [&output];
        output.push_forward_instruction_f32(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
        ); //
        output.push_forward_instruction_f32(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
        ); //
        output.push_forward_instruction_f32(
            Rc::new(self.clone()),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        );
        let inputs = [input_1, input_2];
        let outputs = [input_2];
        output.push_backward_instruction(
            Rc::new(CrossEntropyLossBackward::default()),
            &inputs,  //
            &outputs, //
        );
        Ok(output)
    }
}

impl Operator for CrossEntropyLoss {
    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        self.forward_f32(
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        )
    }

    fn forward_f32(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let expected = inputs[0];
        let actual = inputs[1];
        let loss = CrossEntropyLoss::evaluate(&self.device, expected, actual)?;
        outputs[0].set_values(vec![loss; 1]);
        Ok(())
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

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        self.forward_f32(
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].gradient().deref().borrow()],
        )
    }

    fn forward_f32(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let expected = inputs[0];
            let actual = inputs[1];
            CrossEntropyLoss::derive(expected, actual, output_gradient)?;
        }
        Ok(())
    }
}
