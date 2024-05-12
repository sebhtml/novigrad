use std::{ops::Deref, rc::Rc};

use crate::{BinaryOperator, Device, Instruction, Operator, Tensor, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Add.html
#[derive(Clone)]
pub struct Add {
    device: Device,
}

impl Add {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.to_owned(),
        }
    }
}

impl BinaryOperator for Add {
    fn forward(&self, input_1: &Tensor, input_2: &Tensor) -> Result<Tensor, crate::Error> {
        let input_0_t: &TensorF32 = &input_1.tensor().deref().borrow();
        let input_1_t: &TensorF32 = &input_1.tensor().deref().borrow();
        debug_assert_eq!(input_0_t.size(), input_1_t.size());
        let rows = input_0_t.rows();
        let cols = input_0_t.cols();
        let len = rows * cols;
        let output = self.device.tensor(rows, cols, vec![0.0; len], true, false);
        output.push_forward_instruction(Instruction::new(
            Rc::new(self.clone()),
            &[input_1, input_2],
            &[&output],
        ));
        Ok(output)
    }
}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), crate::Error> {
        let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut outputs[0].tensor().deref().borrow_mut();
        TensorF32::copy(input_0, output)?;
        TensorF32::add(input_1, output)
    }

    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), crate::Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();

        if inputs[1].requires_grad() {
            let input_1_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
            TensorF32::copy(output_gradient, input_1_gradient)?;
        }

        if inputs[0].requires_grad() {
            let input_0_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            TensorF32::copy(output_gradient, input_0_gradient)?;
        }

        Ok(())
    }
}
