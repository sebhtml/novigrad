use std::{cell::RefCell, rc::Rc};

use crate::Tensor;

#[derive(Clone)]
pub struct LearningTensor {
    tensor: Rc<RefCell<Tensor>>,
    gradient: Rc<RefCell<Tensor>>,
}

impl LearningTensor {
    pub fn new(tensor: Rc<RefCell<Tensor>>, gradient: Rc<RefCell<Tensor>>) -> Self {
        Self { tensor, gradient }
    }
    pub fn tensor(&self) -> &Rc<RefCell<Tensor>> {
        &self.tensor
    }
    pub fn gradient(&self) -> &Rc<RefCell<Tensor>> {
        &self.gradient
    }
}
