use std::{ops::Deref, rc::Rc};

use crate::{Device, Instruction, Operator, Tensor, TensorF32, UnaryOperator};

/// Linear is not a ONNX operator. https://onnx.ai/onnx/operators/index.html ???
/// TODO implement broadcasting to use Mul instead
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
        let output = self.device.tensor(rows, cols, vec![0.0; len], true, false);
        let inputs = &[input];
        let outputs = &[&output];
        output.push_forward_instruction(Rc::new(self.clone()), inputs, outputs);
        output.push_backward_instruction(Rc::new(ScaleBackward::new(self.alpha)), outputs, inputs);
        Ok(output)
    }
}

impl Operator for Scale {
    fn name(&self) -> &str {
        "Scale"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), crate::Error> {
        let input = &inputs[0].tensor().deref().borrow();
        let output = &outputs[0].tensor().deref().borrow();
        TensorF32::copy(input, output)?;
        let alpha = self.alpha;
        TensorF32::scale(alpha, output)
    }
}

pub struct ScaleBackward {
    alpha: f32,
}

impl ScaleBackward {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Operator for ScaleBackward {
    fn name(&self) -> &str {
        "ScaleBackward"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), crate::Error> {
        debug_assert_eq!(outputs.len(), 1);
        let input_gradient: &TensorF32 = &inputs[0].gradient().deref().borrow();

        if outputs[0].requires_grad() {
            let output_gradient: &mut TensorF32 = &mut outputs[0].gradient().deref().borrow_mut();
            TensorF32::copy(input_gradient, output_gradient)?;
            // TODO this looks wrong.
            let alpha = self.alpha;
            TensorF32::scale(alpha, output_gradient)?;
        }

        Ok(())
    }
}
