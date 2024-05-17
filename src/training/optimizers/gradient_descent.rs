use std::{ops::Deref, rc::Rc};

use crate::{Add, Device, Error, Instruction, OptimizerTrait, Scale, Tensor, TensorF32};

pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl OptimizerTrait for GradientDescent {
    fn optimize(&self, device: &Device, tensors: &[Tensor]) -> Result<Vec<Instruction>, Error> {
        let mut instructions = vec![];
        for optimizable_tensor in tensors {
            let tensor: &TensorF32 = &optimizable_tensor.tensor().deref().borrow();
            let gradient: &TensorF32 = &optimizable_tensor.gradient().deref().borrow();
            debug_assert_eq!(gradient.size(), tensor.size(),);
            let tmp = device.tensor_f32(tensor.rows(), tensor.cols(), vec![0.0; tensor.len()]);
            TensorF32::scale(0.0, &tmp)?;
            instructions.push(Instruction::new(
                Rc::new(Scale::new(device, 0.0)),
                &[&tmp],
                &[&tmp],
                false,
            ));
            TensorF32::add(gradient, &tmp)?;
            instructions.push(Instruction::new(
                Rc::new(Add::new(device)),
                &[gradient, &tmp],
                &[&tmp],
                false,
            ));
            TensorF32::scale(-self.learning_rate, &tmp)?;
            instructions.push(Instruction::new(
                Rc::new(Scale::new(device, -self.learning_rate)),
                &[&tmp],
                &[&tmp],
                false,
            ));
            TensorF32::add(&tmp, tensor)?;
            instructions.push(Instruction::new(
                Rc::new(Add::new(device)),
                &[&tmp, tensor],
                &[tensor],
                false,
            ));
        }
        println!(
            "GradientDescent: Generated {} instructions for {} optimizable tensors",
            instructions.len(),
            tensors.len()
        );
        Ok(instructions)
    }
}
