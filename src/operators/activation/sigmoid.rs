use crate::devices::Device;
use crate::{instruction, Error, Tensor};
use crate::{Instruction, OpCode, Operator, TensorF32, UnaryOperator};
use std::f32::consts::E;
use std::ops::Deref;
use std::rc::Rc;

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
        output.push_instruction(instruction!(
            OpCode::Scale(0.0),
            &[&outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
            crate::Category::Inference,
        ));
        output.push_instruction(instruction!(
            OpCode::Scale(0.0),
            &[&outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
            crate::Category::Inference,
        ));
        output.push_instruction(instruction!(
            OpCode::DynOperator(Rc::new(self.clone())),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
            crate::Category::Inference,
        ));
        let inputs = [&output];
        let outputs = [input];
        output.push_instruction(instruction!(
            OpCode::DynOperator(Rc::new(SigmoidBackward::new(&self.device))),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[0].gradient().deref().borrow(),
                &outputs[0].tensor().deref().borrow(),
            ],
            &[&outputs[0].gradient().deref().borrow()],
            crate::Category::Gradient,
        ));
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
            let one_minus_output = self.device.tensor_f32(rows, cols, vec![1.0; len]);
            TensorF32::sub(input, &one_minus_output)?;
            let layer_f_derivative = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            TensorF32::mul(input, &one_minus_output, &layer_f_derivative)?;
            let mut tmp = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            TensorF32::mul(&layer_f_derivative, input_gradient, &mut tmp)?;
            TensorF32::add(&tmp, output_gradient)?;
        }

        Ok(())
    }
}
