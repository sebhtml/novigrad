use crate::{
    devices::Device,
    error, instruction, new_tensor, new_tensor_with_grad,
    opcode::OpCode,
    tensor::{Error, ErrorEnum, Tensor},
    BinaryOperator, Category, OperatorAttributes, TensorWithGrad,
};

pub struct MatMul {
    device: Device,
    transb: bool,
}

impl MatMul {
    pub fn new(device: &Device, transb: bool) -> Self {
        MatMul {
            device: device.clone(),
            transb,
        }
    }
}

impl BinaryOperator for MatMul {
    fn forward(
        &self,
        input_0: &TensorWithGrad,
        input_1: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error> {
        let input_0_tensor: &Tensor = &input_0.tensor();
        let input_1_tensor: &Tensor = &input_1.tensor();
        let compatible = match self.transb {
            false => input_0_tensor.cols() == input_1_tensor.rows(),
            true => input_0_tensor.cols() == input_1_tensor.cols(),
        };
        if !compatible {
            println!("Incompatible shapes in matrix multiplication");
            println!("transa: false, transb: {}", self.transb);
            println!(
                "Between A {:?} and B {:?}",
                *input_0_tensor.size(),
                *input_1_tensor.size(),
            );
            debug_assert!(false);
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }

        let rows = input_0_tensor.rows();
        let transb = self.transb;
        let cols = if transb {
            input_1_tensor.rows()
        } else {
            input_1_tensor.cols()
        };
        let len = rows * cols;
        let output = new_tensor_with_grad!(
            self.device,
            rows,
            cols,
            vec![0.0; len],
            &[input_0, input_1],
            true,
            false,
        )?;

        let inputs = [input_0, input_1];
        let outputs = [&output];

        // For MatMul, we need to zero C it uses Gemm and
        // Gemm is C := alpha * AB^T + beta * C
        let zero = new_tensor!(&self.device, 1, 1, vec![0.0])?;
        output.push_instruction(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&zero, &outputs[0].tensor()],
            &[&outputs[0].tensor()],
            Category::Inference,
        ));

        output.push_instruction(instruction!(
            OpCode::Gemm,
            OperatorAttributes::ThreeBools(false, transb, false),
            &[
                &inputs[0].tensor(),
                &inputs[1].tensor(),
                &outputs[0].tensor(),
            ],
            &[&outputs[0].tensor()],
            Category::Inference,
        ));
        if input_1.gradient().requires_grad() {
            output.push_instruction(instruction!(
                OpCode::Gemm,
                OperatorAttributes::ThreeBools(true, false, transb),
                &[&input_0.tensor(), &output.gradient(), &input_1.gradient(),],
                &[&input_1.gradient()],
                Category::Gradient,
            ));
        }

        if input_0.gradient().requires_grad() {
            output.push_instruction(instruction!(
                OpCode::Gemm,
                OperatorAttributes::ThreeBools(false, !transb, false),
                &[&output.gradient(), &input_1.tensor(), &input_0.gradient(),],
                &[&input_0.gradient()],
                Category::Gradient,
            ));
        }

        Ok(output)
    }
}
