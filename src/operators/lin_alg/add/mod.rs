use crate::{
    error, instruction, new_tensor_with_grad,
    opcode::OpCode,
    stream::DeviceStream,
    tensor::{Error, ErrorEnum, Tensor},
    BinaryOperator, Category, Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
    TensorWithGrad,
};

#[cfg(test)]
mod tests;

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

impl ExecutableOperator for Add {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        if input_0.len() != input_1.len() {
            println!("Incompatible sizes");
            println!("x {}", input_0);
            println!("y {}", input_1);
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        if input_0.len() != output.len() {
            println!("Incompatible sizes");
            println!("x {}", input_0);
            println!("y {}", output);
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        device.copy_to(input_0, output, device_stream)?;

        let alpha = &device_stream.one;

        let n = input_1.len() as i32;
        let incx = 1;
        let incy = 1;
        device.axpy(n, alpha, input_1, incx, output, incy, device_stream)
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

        output.push_instruction(instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&inputs[0].tensor(), &inputs[1].tensor(),],
            &[&outputs[0].tensor()],
            Category::Inference,
        ));

        {
            let inputs = [&output];
            let outputs = [input_1, input_2];

            let inputs = &[&inputs[0].gradient()];
            let outputs = &[&outputs[0].gradient(), &outputs[1].gradient()];

            let input_gradient = inputs[0];

            if outputs[1].requires_grad() {
                let output_1_gradient = outputs[1];

                output.push_instruction(instruction!(
                    OpCode::Add,
                    OperatorAttributes::None,
                    &[input_gradient, output_1_gradient],
                    &[output_1_gradient],
                    Category::Gradient,
                ));
            }

            if outputs[0].requires_grad() {
                let output_0_gradient = outputs[0];

                output.push_instruction(instruction!(
                    OpCode::Add,
                    OperatorAttributes::None,
                    &[input_gradient, output_0_gradient],
                    &[output_0_gradient],
                    Category::Gradient,
                ));
            }
        }

        Ok(output)
    }
}
