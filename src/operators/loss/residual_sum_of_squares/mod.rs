use std::{ops::Deref, rc::Rc};

use crate::{devices::Device, BinaryOperator, Error, ErrorEnum, Operator, Tensor, TensorF32, Zero};

use super::LossFunction;

#[cfg(test)]
mod tests;

/// ResidualSumOfSquares is not a ONNX operator.
/// https://onnx.ai/onnx/operators/index.html ???
#[derive(Clone)]
pub struct ResidualSumOfSquares {
    device: Device,
}

impl ResidualSumOfSquares {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl LossFunction for ResidualSumOfSquares {
    /// RSS = Σ (y_i - f(x_i))^2
    fn evaluate(device: &Device, expected: &TensorF32, actual: &TensorF32) -> Result<f32, Error> {
        if expected.size() != actual.size() {
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }
        let rows = expected.rows();
        let cols = expected.cols();
        let len = rows * cols;
        let mut diffs = device.tensor_f32(rows, cols, vec![0.0; len]);
        TensorF32::copy(expected, &mut diffs)?;
        TensorF32::sub(actual, &mut diffs)?;
        TensorF32::dot_product(&diffs, &diffs)
    }

    fn derive(expected: &TensorF32, actual: &TensorF32, result: &TensorF32) -> Result<(), Error> {
        TensorF32::copy(expected, result)?;
        TensorF32::sub(actual, result)?;
        TensorF32::scale(-2.0, result)
    }
}

impl BinaryOperator for ResidualSumOfSquares {
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
        );
        output.push_forward_instruction_f32(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
        );
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
            Rc::new(ResidualSumOfSquaresBackward::default()),
            &inputs,  //
            &outputs, //
        );
        Ok(output)
    }
}

impl Operator for ResidualSumOfSquares {
    fn name(&self) -> &str {
        "ResidualSumOfSquares"
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
        let loss = ResidualSumOfSquares::evaluate(&self.device, expected, actual)?;
        outputs[0].set_values(vec![loss; 1]);
        Ok(())
    }
}

pub struct ResidualSumOfSquaresBackward {}

impl Default for ResidualSumOfSquaresBackward {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for ResidualSumOfSquaresBackward {
    fn name(&self) -> &str {
        "ResidualSumOfSquaresBackward"
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
            ResidualSumOfSquares::derive(expected, actual, output_gradient)?;
        }
        Ok(())
    }
}
