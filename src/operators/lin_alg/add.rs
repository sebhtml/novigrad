use crate::{
    gradient_instruction, inference_instruction, new_tensor, new_tensor_with_grad,
    tensor::{Error, Tensor},
    BinaryOperator, Device, OpCode, TensorWithGrad,
};

pub struct Add {
    device: Device,
}

impl Add {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.to_owned(),
        }
    }

    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        Tensor::copy(input_0, output)?;
        Tensor::add(input_1, output)
    }
}

impl BinaryOperator for Add {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let input_0_t: &Tensor = &input_1.tensor();
        let input_1_t: &Tensor = &input_1.tensor();
        debug_assert_eq!(*input_0_t.size(), *input_1_t.size());
        let rows = input_0_t.rows();
        let cols = input_0_t.cols();
        let len = rows * cols;
        let output = new_tensor_with_grad!(
            self.device,
            rows,
            cols,
            vec![0.0; len],
            &[input_1, input_2],
            true,
            false,
        )?;
        let inputs = [input_1, input_2];
        let outputs = [&output];
        let zero = new_tensor!(self.device, 1, 1, vec![0.0])?;
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].tensor()],
            &[&outputs[0].tensor()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul,
            &[&zero, &outputs[0].gradient()],
            &[&outputs[0].gradient()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::Add,
            &[&inputs[0].tensor(), &inputs[1].tensor(),],
            &[&outputs[0].tensor()],
        ));

        {
            let inputs = [&output];
            let outputs = [input_1, input_2];

            let inputs = &[&inputs[0].gradient()];
            let outputs = &[&outputs[0].gradient(), &outputs[1].gradient()];

            let input_gradient = inputs[0];

            if outputs[1].requires_grad() {
                let output_1_gradient = outputs[1];

                output.push_instruction(gradient_instruction!(
                    OpCode::Add,
                    &[input_gradient, output_1_gradient],
                    &[output_1_gradient],
                ));
            }

            if outputs[0].requires_grad() {
                let output_0_gradient = outputs[0];

                output.push_instruction(gradient_instruction!(
                    OpCode::Add,
                    &[input_gradient, output_0_gradient],
                    &[output_0_gradient],
                ));
            }
        }

        Ok(output)
    }
}
