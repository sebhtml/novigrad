use std::{ops::Deref, rc::Rc};

use crate::{Error, Operator, Tensor};

#[derive(Clone, Debug)]
pub struct Instruction {
    operator: Rc<dyn Operator>,
    inputs: Rc<Vec<Tensor>>,
    outputs: Rc<Vec<Tensor>>,
}

impl Instruction {
    pub fn new(operator: Rc<dyn Operator>, inputs: &[&Tensor], outputs: &[&Tensor]) -> Self {
        let inputs: Vec<Tensor> = inputs.into_iter().map(|x| (*x).clone()).collect();
        let outputs: Vec<Tensor> = outputs.into_iter().map(|x| (*x).clone()).collect();
        Self {
            operator,
            inputs: inputs.into(),
            outputs: outputs.into(),
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
        self.operator.forward(&inputs, &outputs)
    }

    pub fn backward(&self) -> Result<(), Error> {
        let inputs: Vec<&Tensor> = self.inputs.iter().collect();
        let outputs: Vec<&Tensor> = self.outputs.iter().collect();
        self.operator.backward(&inputs, &outputs)
    }
}
