use crate::{
    instruction, new_tensor, opcode::OpCode, tensor::Error, Category, Device, Instruction,
    OperatorAttributes, OptimizerTrait, TensorWithGrad,
};

pub struct StochasticGradientDescent {
    learning_rate: f32,
}

impl StochasticGradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl OptimizerTrait for StochasticGradientDescent {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error> {
        let mut instructions = vec![];
        for optimizable_tensor in tensors {
            let tensor = &optimizable_tensor.tensor();
            let gradient = &optimizable_tensor.gradient();
            debug_assert_eq!(*gradient.size(), *tensor.size(),);

            let scaled_gradient = new_tensor!(
                device,
                tensor.rows(),
                tensor.cols(),
                vec![0.0; tensor.len()]
            )?;

            let alpha = new_tensor!(device, 1, 1, vec![self.learning_rate])?;
            instructions.push(instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&alpha, &gradient],
                &[&scaled_gradient],
                Category::Optimization,
            ));

            instructions.push(instruction!(
                OpCode::Sub,
                OperatorAttributes::None,
                &[tensor, &scaled_gradient],
                &[tensor],
                Category::Optimization,
            ));
        }

        Ok(instructions)
    }
}
