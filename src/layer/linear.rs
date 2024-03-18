use std::{cell::RefCell, rc::Rc};

use crate::{ActivationFunction, Layer, Tensor};

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
}
