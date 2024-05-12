use std::{ops::Deref, rc::Rc};

use crate::{Device, Operator, Tensor, TensorF32, UnaryOperator};

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
        let output = self.device.tensor(
            rows,
            cols,
            vec![0.0; len],
            Rc::new(self.clone()),
            &[input],
            true,
            false,
        );
        Ok(output)
    }
}

impl Operator for Scale {
    fn name(&self) -> &str {
        "Scale"
    }

    fn forward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), crate::Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        TensorF32::copy(input, output)?;
        let alpha = self.alpha;
        TensorF32::scale(alpha, output)
    }

    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), crate::Error> {
        debug_assert_eq!(inputs.len(), 1);
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();

        if inputs[0].requires_grad() {
            let input_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            TensorF32::copy(output_gradient, input_gradient)?;
            let alpha = self.alpha;
            TensorF32::scale(alpha, input_gradient)?;
        }

        Ok(())
    }
}
