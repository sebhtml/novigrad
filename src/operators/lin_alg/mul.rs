use std::{ops::Deref, rc::Rc};

use crate::{BinaryOperator, Device, Error, Operator, Tensor, TensorF32, Zero};

/// https://onnx.ai/onnx/operators/onnx__Mul.html
#[derive(Clone)]
pub struct Mul {
    device: Device,
}

impl Mul {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl BinaryOperator for Mul {
    fn forward(&self, input_0: &Tensor, input_1: &Tensor) -> Result<Tensor, Error> {
        let input_0_t: &TensorF32 = &input_0.tensor().deref().borrow();
        let input_1_t: &TensorF32 = &input_1.tensor().deref().borrow();
        debug_assert_eq!(input_0_t.size(), input_1_t.size());
        let rows = input_0_t.rows();
        let cols = input_0_t.cols();
        let len = rows * cols;
        let output =
            self.device
                .tensor(rows, cols, vec![0.0; len], &[input_0, input_1], true, false);
        let inputs = [input_0, input_1];
        let outputs = [&output];
        output.push_forward_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        output.push_forward_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
            false,
        );
        output.push_forward_instruction(
            Rc::new(self.clone()),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        let inputs = [input_0, input_1, &output];
        let outputs = [input_0, input_1];
        output.push_backward_instruction(
            Rc::new(MulBackward::new(&self.device)),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
                &inputs[2].gradient().deref().borrow(),
            ],
            &[
                &outputs[0].gradient().deref().borrow(),
                &outputs[1].gradient().deref().borrow(),
            ],
        );
        Ok(output)
    }
}

impl Operator for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        TensorF32::mul(input_0, input_1, output)
    }
}

pub struct MulBackward {
    device: Device,
}

impl MulBackward {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl Operator for MulBackward {
    fn name(&self) -> &str {
        "MulBackward"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        debug_assert_eq!(outputs.len(), 2);
        let input_gradient = inputs[2];
        let rows = input_gradient.rows();
        let cols = input_gradient.cols();
        let len = rows * cols;

        if outputs[1].requires_grad() {
            let output_1_gradient = outputs[1];
            let output_0 = inputs[0];
            let mut tmp = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            TensorF32::mul(output_0, input_gradient, &mut tmp)?;
            TensorF32::add(&tmp, output_1_gradient)?;
        }

        if outputs[0].requires_grad() {
            let output_0_gradient = outputs[0];
            let output = inputs[1];
            let mut tmp = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            TensorF32::mul(output, input_gradient, &mut tmp)?;
            TensorF32::add(&tmp, output_0_gradient)?;
        }

        Ok(())
    }
}
