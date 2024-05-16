use std::rc::Rc;

use crate::{Error, Operator, TensorF32};

#[derive(Clone, Debug)]
pub struct Instruction {
    operator: Rc<dyn Operator>,
    inputs_f32: Rc<Vec<TensorF32>>,
    outputs_f32: Rc<Vec<TensorF32>>,
}

impl Instruction {
    pub fn new_f32(
        operator: Rc<dyn Operator>,
        inputs_f32: &[&TensorF32],
        outputs_f32: &[&TensorF32],
    ) -> Self {
        let inputs_f32: Vec<TensorF32> = inputs_f32.into_iter().map(|x| (*x).clone()).collect();
        let outputs_f32: Vec<TensorF32> = outputs_f32.into_iter().map(|x| (*x).clone()).collect();
        Self {
            operator,
            inputs_f32: inputs_f32.into(),
            outputs_f32: outputs_f32.into(),
        }
    }

    pub fn operator(&self) -> &Rc<dyn Operator> {
        &self.operator
    }
    pub fn inputs_f32(&self) -> &Rc<Vec<TensorF32>> {
        &self.inputs_f32
    }
    pub fn outputs_f32(&self) -> &Rc<Vec<TensorF32>> {
        &self.outputs_f32
    }
    pub fn forward_f32(&self) -> Result<(), Error> {
        let inputs_f32: Vec<&TensorF32> = self.inputs_f32.iter().collect();
        let outputs_f32: Vec<&TensorF32> = self.outputs_f32.iter().collect();
        if inputs_f32.len() > 0 || outputs_f32.len() > 0 {
            self.operator.forward_f32(&inputs_f32, &outputs_f32)?;
        }
        Ok(())
    }
}
