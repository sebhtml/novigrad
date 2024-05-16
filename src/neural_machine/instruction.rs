use std::{ops::Deref, rc::Rc};

use crate::{Error, Operator, Tensor, TensorF32};

#[derive(Clone, Debug)]
pub struct Instruction {
    operator: Rc<dyn Operator>,
    inputs: Rc<Vec<Tensor>>,
    outputs: Rc<Vec<Tensor>>,
    inputs_f32: Rc<Vec<TensorF32>>,
    outputs_f32: Rc<Vec<TensorF32>>,
}

impl Instruction {
    pub fn new(operator: Rc<dyn Operator>, inputs: &[&Tensor], outputs: &[&Tensor]) -> Self {
        let inputs: Vec<Tensor> = inputs.into_iter().map(|x| (*x).clone()).collect();
        let outputs: Vec<Tensor> = outputs.into_iter().map(|x| (*x).clone()).collect();
        Self {
            operator,
            inputs: inputs.into(),
            outputs: outputs.into(),
            inputs_f32: Default::default(),
            outputs_f32: Default::default(),
        }
    }
    pub fn new_f32(
        operator: Rc<dyn Operator>,
        inputs_f32: &[&TensorF32],
        outputs_f32: &[&TensorF32],
    ) -> Self {
        let inputs_f32: Vec<TensorF32> = inputs_f32.into_iter().map(|x| (*x).clone()).collect();
        let outputs_f32: Vec<TensorF32> = outputs_f32.into_iter().map(|x| (*x).clone()).collect();
        Self {
            operator,
            inputs: Default::default(),
            outputs: Default::default(),
            inputs_f32: inputs_f32.into(),
            outputs_f32: outputs_f32.into(),
        }
    }

    pub fn operator(&self) -> &Rc<dyn Operator> {
        &self.operator
    }
    pub fn inputs(&self) -> &Rc<Vec<Tensor>> {
        &self.inputs
    }
    pub fn outputs(&self) -> &Rc<Vec<Tensor>> {
        &self.outputs
    }
    pub fn forward(&self) -> Result<(), Error> {
        let inputs: Vec<&Tensor> = self.inputs.iter().collect();
        let outputs: Vec<&Tensor> = self.outputs.iter().collect();
        if inputs.len() > 0 || outputs.len() > 0 {
            self.operator.forward(&inputs, &outputs)?;
        }
        Ok(())
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
