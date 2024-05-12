use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, Instruction, Operator, Tensor, TensorF32, UnaryOperator};

/// https://onnx.ai/onnx/operators/onnx__Identity.html
#[derive(Clone)]
pub struct Identity {
    device: Device,
}

impl Identity {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl UnaryOperator for Identity {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_t: &TensorF32 = &input.tensor().deref().borrow();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output = self.device.tensor(rows, cols, vec![0.0; len], true, false);
        output.push_forward_instruction(Instruction::new(
            Rc::new(self.clone()),
            &[input],
            &[&output],
        ));
        Ok(output)
    }
}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::copy(input, output)
    }

    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        if inputs[0].requires_grad() {
            let input_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
            TensorF32::copy(output_gradient, input_gradient)?;
        }

        Ok(())
    }
}
