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
        let inputs = &[input];
        let outputs = &[&output];
        output.push_forward_instruction(Rc::new(self.clone()), inputs, outputs);
        Ok(output)
    }
}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0].tensor().deref().borrow();
        let output = outputs[0].tensor().deref().borrow();
        TensorF32::copy(&input, &output)
    }

    fn backward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let instruction = Instruction::new(Rc::new(IdentityBackward::default()), inputs, outputs);
        instruction.forward()
    }
}

pub struct IdentityBackward {}

impl Default for IdentityBackward {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for IdentityBackward {
    fn name(&self) -> &str {
        "IdentityBackward"
    }

    // TODO inverse inputs and outputs
    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        if inputs[0].requires_grad() {
            let input_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let output_gradient: &TensorF32 = &outputs[0].gradient().deref().borrow();
            TensorF32::copy(output_gradient, input_gradient)?;
        }

        Ok(())
    }

    fn backward(&self, _inputs: &[&Tensor], _outputs: &[&Tensor]) -> Result<(), Error> {
        todo!()
    }
}
