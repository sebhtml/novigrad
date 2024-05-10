use std::{ops::Deref, rc::Rc};

use crate::{devices::Device, Error, Operator, Tensor, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Reshape.html
#[derive(Clone)]
pub struct Reshape {
    device: Device,
    input_size: Vec<usize>,
    output_size: Vec<usize>,
}

impl Reshape {
    pub fn new(device: &Device, input_size: Vec<usize>, output_size: Vec<usize>) -> Self {
        Self {
            device: device.clone(),
            input_size,
            output_size,
        }
    }
}

impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 1);
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        debug_assert_eq!(input.size(), self.input_size);
        let rows = self.output_size[0];
        let cols = self.output_size[1];
        let len = rows * cols;
        let output = self.device.tensor(
            Rc::new(self.clone()),
            inputs,
            rows,
            cols,
            vec![0.0; len],
            true,
            false,
        );
        Ok(output)
    }

    fn forward_realize(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        TensorF32::copy(input, output)?;
        output.resize(&self.output_size)
    }

    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        if inputs[0].requires_grad() {
            let input_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
            TensorF32::copy(output_gradient, input_gradient)?;
            input_gradient.resize(&self.input_size)?;
        }
        Ok(())
    }
}
