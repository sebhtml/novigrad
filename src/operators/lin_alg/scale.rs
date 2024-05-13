use std::{ops::Deref, rc::Rc};

use crate::{Device, Identity, Operator, Tensor, TensorF32, UnaryOperator};

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
        let output = self
            .device
            .tensor(rows, cols, vec![0.0; len], &[input], true, false);
        output.push_forward_instruction(Rc::new(self.clone()), &[input], &[&output]);
        output.push_backward_instruction(
            Rc::new(Identity::new(&self.device)),
            &[&output],
            &[input],
        );
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
