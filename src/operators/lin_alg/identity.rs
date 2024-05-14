use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, Operator, Tensor, TensorF32, UnaryOperator};

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
        let output = self
            .device
            .tensor(rows, cols, vec![0.0; len], &[input], true, false);
        output.push_forward_instruction(Rc::new(self.clone()), &[input], &[&output]);
        output.push_backward_instruction(
            Rc::new(IdentityBackward::default()),
            &[&output],
            &[input],
        );
        Ok(output)
    }
}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        self.forward_f32(
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        )
    }

    fn forward_f32(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::copy(&input, &output)
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

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        self.forward_f32(
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        )
    }

    fn forward_f32(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        if outputs[0].requires_grad() {
            let output_gradient = outputs[0];
            let input_gradient = inputs[0];
            TensorF32::copy(input_gradient, output_gradient)?;
        }

        Ok(())
    }
}
