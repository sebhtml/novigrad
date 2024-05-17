use std::{ops::Deref, rc::Rc};

use crate::{BinaryOperator, Device, Error, Operator, Tensor, TensorF32, Zero};

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
    fn forward(&self, input_1: &Tensor, input_2: &Tensor) -> Result<Tensor, Error> {
        let input_0_t: &TensorF32 = &input_1.tensor().deref().borrow();
        let input_1_t: &TensorF32 = &input_1.tensor().deref().borrow();
        debug_assert_eq!(input_0_t.size(), input_1_t.size());
        let rows = input_0_t.rows();
        let cols = input_0_t.cols();
        let len = rows * cols;
        let output =
            self.device
                .tensor(rows, cols, vec![0.0; len], &[input_1, input_2], true, false);
        let inputs = [input_1, input_2];
        let outputs = [&output];
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(self.clone()),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        let inputs = [&output];
        let outputs = [input_1, input_2];
        output.push_backward_instruction(
            Rc::new(AddBackward::new()),
            &[&inputs[0].gradient().deref().borrow()],
            &[
                &outputs[0].gradient().deref().borrow(),
                &outputs[1].gradient().deref().borrow(),
            ],
            true,
        );
        Ok(output)
    }
}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        TensorF32::copy(input_0, output)?;
        TensorF32::add(input_1, output)
    }
}

pub struct AddBackward {}

impl AddBackward {
    pub fn new() -> Self {
        Self {}
    }
}
impl Operator for AddBackward {
    fn name(&self) -> &str {
        "AddBackward"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        debug_assert_eq!(outputs.len(), 2);
        let input_gradient = inputs[0];

        if outputs[1].requires_grad() {
            let output_1_gradient = outputs[1];
            TensorF32::add(input_gradient, output_1_gradient)?;
        }

        if outputs[0].requires_grad() {
            let output_0_gradient = outputs[0];
            TensorF32::add(input_gradient, output_0_gradient)?;
        }

        Ok(())
    }
}
