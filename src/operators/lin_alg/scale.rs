use std::{ops::Deref, rc::Rc};

use crate::{Device, Operator, Tensor, TensorF32, UnaryOperator, Zero};

/// Scale is not a ONNX operator. https://onnx.ai/onnx/operators/index.html ???
#[derive(Clone)]
pub struct Scale {
    device: Device,
    alpha: f32,
}

impl Scale {
    pub fn new(device: &Device, alpha: f32) -> Self {
        Self {
            device: device.clone(),
            alpha,
        }
    }
}

impl UnaryOperator for Scale {
    fn forward(&self, input: &Tensor) -> Result<Tensor, crate::Error> {
        let input_t: &TensorF32 = &input.tensor().deref().borrow();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output = self
            .device
            .tensor(rows, cols, vec![0.0; len], &[input], true, false);
        let inputs = [input];
        let outputs = [&output];
        output.push_forward_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
        );
        output.push_forward_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
        );
        output.push_forward_instruction(
            Rc::new(self.clone()),
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        );
        let inputs = [&output];
        let outputs = [input];
        output.push_backward_instruction_f32(
            Rc::new(ScaleBackward::default()),
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        );
        Ok(output)
    }
}

impl Operator for Scale {
    fn name(&self) -> &str {
        "Scale"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), crate::Error> {
        self.forward_f32(
            &[&inputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        )
    }

    fn forward_f32(
        &self,
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
    ) -> Result<(), crate::Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::copy(input, output)?;
        let alpha = self.alpha;
        TensorF32::scale(alpha, output)
    }
}

pub struct ScaleBackward {}

impl Default for ScaleBackward {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for ScaleBackward {
    fn name(&self) -> &str {
        "ScaleBackward"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), crate::Error> {
        self.forward_f32(
            &[&inputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        )
    }

    fn forward_f32(
        &self,
        inputs: &[&TensorF32],
        outputs: &[&TensorF32],
    ) -> Result<(), crate::Error> {
        let input = inputs[0];
        let output = outputs[0];
        TensorF32::add(input, output)
    }
}
