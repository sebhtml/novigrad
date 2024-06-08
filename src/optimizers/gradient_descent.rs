use crate::{
    new_tensor, optimization_instruction, tensor::Error, Device, Instruction, OpCode,
    OptimizerTrait, TensorWithGrad,
};

pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl OptimizerTrait for GradientDescent {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error> {
        let mut instructions = vec![];
        let zero = new_tensor!(device, 1, 1, vec![0.0])?;
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
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                &[&zero, &scaled_gradient],
                &[&scaled_gradient],
            ));

            instructions.push(optimization_instruction!(
                OpCode::Add,
                &[&scaled_gradient, gradient],
                &[&scaled_gradient],
            ));

            let alpha = new_tensor!(device, 1, 1, vec![-self.learning_rate])?;
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                &[&alpha, &scaled_gradient],
                &[&scaled_gradient],
            ));

            instructions.push(optimization_instruction!(
                OpCode::Add,
                &[tensor, &scaled_gradient],
                &[tensor],
            ));
        }

        Ok(instructions)
    }
}
