use std::{ops::Deref, rc::Rc};

use crate::{devices::Device, Error, Instruction, Operator, Tensor, TensorF32, UnaryOperator};

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

impl UnaryOperator for Reshape {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_tensor: &TensorF32 = &input.tensor().deref().borrow();
        debug_assert_eq!(*input_tensor.size().deref().borrow_mut(), self.input_size);
        let rows = self.output_size[0];
        let cols = self.output_size[1];
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

impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input = &inputs[0];
        let output = outputs[0];
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
