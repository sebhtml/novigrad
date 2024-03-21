use std::{cell::RefCell, rc::Rc};

use crate::{ActivationFunction, Error, Layer, Tensor};

pub struct Linear {
    pub weights: Rc<RefCell<Tensor>>,
    pub activation: Rc<dyn ActivationFunction>,
}

impl Layer for Linear {
    fn weights(&self) -> Rc<RefCell<Tensor>> {
        self.weights.clone()
    }

    fn activation(&self) -> Rc<dyn ActivationFunction> {
        self.activation.clone()
    }

    fn forward(&self, input: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        input.matmul(&(*self.weights.borrow()).transpose(), result)
    }
}
