use std::ops::Deref;

use crate::{
    instruction, BinaryOperator, Category, Device, Error, Instruction, OpCode, Tensor, TensorF32,
};

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

    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        TensorF32::mul(input_0, input_1, output)
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
        output.push_instruction(instruction!(
            OpCode::Scale(0.0),
            &[&outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
            crate::Category::Inference,
        ));
        output.push_instruction(instruction!(
            OpCode::Scale(0.0),
            &[&outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
            crate::Category::Inference,
        ));
        output.push_instruction(instruction!(
            OpCode::Mul,
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
            crate::Category::Inference,
        ));

        {
            let inputs = [input_0, input_1, &output];
            let outputs = [input_0, input_1];

            let inputs = &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
                &inputs[2].gradient().deref().borrow(),
            ];

            let outputs = &[
                &outputs[0].gradient().deref().borrow(),
                &outputs[1].gradient().deref().borrow(),
            ];

            debug_assert_eq!(outputs.len(), 2);
            let input_gradient = inputs[2];
            let rows = input_gradient.rows();
            let cols = input_gradient.cols();
            let len = rows * cols;

            if outputs[1].requires_grad() {
                let output_1_gradient = outputs[1];
                let output_0 = inputs[0];
                let mut tmp = self.device.tensor_f32(rows, cols, vec![0.0; len]);

                output.push_instruction(instruction!(
                    OpCode::Mul,
                    &[output_0, input_gradient],
                    &[&mut tmp],
                    Category::Gradient,
                ));

                output.push_instruction(instruction!(
                    OpCode::Add,
                    &[&tmp, output_1_gradient],
                    &[output_1_gradient],
                    Category::Gradient,
                ));
            }

            if outputs[0].requires_grad() {
                let output_0_gradient = outputs[0];
                let output_ = inputs[1];
                let mut tmp = self.device.tensor_f32(rows, cols, vec![0.0; len]);

                output.push_instruction(instruction!(
                    OpCode::Mul,
                    &[output_, input_gradient],
                    &[&mut tmp],
                    Category::Gradient,
                ));

                output.push_instruction(instruction!(
                    OpCode::Add,
                    &[&tmp, output_0_gradient],
                    &[output_0_gradient],
                    Category::Gradient,
                ));
            }
        }

        Ok(output)
    }
}
