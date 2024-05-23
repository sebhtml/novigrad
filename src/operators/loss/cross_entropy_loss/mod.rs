use std::ops::Deref;

use crate::{
    devices::Device, error, gradient_instruction, loss_instruction, BinaryOperator, Error,
    ErrorEnum, OpCode, Tensor, TensorWithGrad, EPSILON,
};

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

    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let expected = inputs[0];
        let actual = inputs[1];
        let loss = CrossEntropyLoss::evaluate(expected, actual)?;
        outputs[0].set_values(vec![loss; 1]);
        Ok(())
    }

    /// H(P, Q) = - Σ (P(i) * log(Q(i)))
    fn evaluate(expected: &Tensor, actual: &Tensor) -> Result<f32, Error> {
        debug_assert_eq!(actual.size(), expected.size());
        let p = expected;
        let q = actual;
        if p.size() != q.size() {
            println!("Incompatible sizes");
            println!("p {}", p);
            println!("q {}", q);
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
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
}

impl BinaryOperator for CrossEntropyLoss {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let output =
            self.device
                .tensor_with_grad(1, 1, vec![0.0], &[input_1, input_2], true, false);
        let inputs = [input_1, input_2];
        let outputs = [&output];
        let zero = self.device.tensor(1, 1, vec![0.0]);
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        output.push_instruction(loss_instruction!(
            OpCode::CrossEntropyLoss,
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [input_1, input_2];
        let outputs = [input_2];

        let inputs: &[&Tensor] = &[
            &inputs[0].tensor().deref().borrow(),
            &inputs[1].tensor().deref().borrow(),
        ];
        let outputs: &[&Tensor] = &[&outputs[0].gradient().deref().borrow()];

        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);

        // When Cross-Entropy Loss is used with a Softmax activation function,
        // then we don't need to derive the softmax activations.
        // The derivative of the Loss in respect to logits (before activation) is
        // output of the softmax function - expected output (one-hot encoded)
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let expected = inputs[0];
            let actual = inputs[1];
            output.push_instruction(gradient_instruction!(
                OpCode::Sub,
                &[actual, expected],
                &[output_gradient],
            ));
        }

        Ok(output)
    }
}
