use std::{ops::Deref, rc::Rc};

use crate::{Error, Operator, Tensor, TensorF32};

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
    pub fn forward(&self) -> Result<(), Error> {
        {
            let output: &Tensor = &self.outputs[0];
            output.tensor().deref().borrow_mut().zero()?;
            output.gradient().deref().borrow_mut().zero()?;
        }
        let inputs: Vec<TensorF32> = self
            .inputs
            .iter()
            .map(|x| x.tensor().deref().borrow().to_owned())
            .collect();
        let outputs: Vec<TensorF32> = self
            .outputs
            .iter()
            .map(|x| x.tensor().deref().borrow().to_owned())
            .collect();
        let inputs: Vec<&TensorF32> = inputs.iter().collect();
        let outputs: Vec<&TensorF32> = outputs.iter().collect();
        self.operator.forward(&inputs, &outputs)
    }

    pub fn backward(&self) -> Result<(), Error> {
        let inputs: Vec<&Tensor> = self.inputs.iter().collect();
        let output: &Tensor = &self.outputs[0];
        self.operator.backward(&inputs, output)
    }
}
