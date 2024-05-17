use crate::devices::Device;
use crate::{ActivationFunction, Operator, TensorF32, UnaryOperator, Zero};
use crate::{Error, Tensor};
use std::f32::consts::E;
use std::ops::Deref;
use std::rc::Rc;

/// https://onnx.ai/onnx/operators/onnx__Sigmoid.html
#[derive(Clone)]
pub struct Sigmoid {
    device: Device,
}

impl Sigmoid {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl ActivationFunction for Sigmoid {
    fn activate(input: &TensorF32, output: &TensorF32) -> Result<(), Error> {
        let rows = input.rows();
        let cols = input.cols();
        let values = input.get_values()?;
        let mut result_values = output.get_values()?;
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = values[input.index(row, col)];
                let y = 1.0 / (1.0 + E.powf(-x));
                result_values[output.index(row, col)] = y;
                col += 1;
            }
            row += 1;
        }
        output.set_values(result_values);
        Ok(())
    }

    fn derive(
        _input: &TensorF32,
        activation_output: &TensorF32,
        output: &mut TensorF32,
    ) -> Result<(), Error> {
        let rows = activation_output.rows();
        let cols = activation_output.cols();
        let values = activation_output.get_values()?;
        let mut result_values = output.get_values()?;
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = values[activation_output.index(row, col)];
                let y = x * (1.0 - x);
                result_values[output.index(row, col)] = y;
                col += 1;
            }
            row += 1;
        }

        output.set_values(result_values);
        Ok(())
    }
}

impl UnaryOperator for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_t: &TensorF32 = &input.tensor().deref().borrow();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output = self
            .device
            .tensor(rows, cols, vec![0.0; len], &[input], true, false);

        let inputs = [input];
        let outputs = [&output];
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(self.clone()),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        let inputs = [&output];
        let outputs = [input];
        output.push_instruction(
            Rc::new(SigmoidBackward::new(&self.device)),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[0].gradient().deref().borrow(),
                &outputs[0].tensor().deref().borrow(),
            ],
            &[&outputs[0].gradient().deref().borrow()],
            true,
        );
        Ok(output)
    }
}

impl Operator for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        Self::activate(&input, &output)
    }
}

pub struct SigmoidBackward {
    device: Device,
}

impl SigmoidBackward {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl Operator for SigmoidBackward {
    fn name(&self) -> &str {
        "SigmoidBackward"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let input_gradient = inputs[1];
            let output = inputs[2];
            let input = inputs[0];
            let rows = output.rows();
            let cols = output.cols();
            let len = rows * cols;
            // Compute activation function derivative.
            let mut layer_f_derivative = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            Sigmoid::derive(output, input, &mut layer_f_derivative)?;
            let mut tmp = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            TensorF32::mul(&layer_f_derivative, input_gradient, &mut tmp)?;
            TensorF32::add(&tmp, output_gradient)?;
        }

        Ok(())
    }
}
