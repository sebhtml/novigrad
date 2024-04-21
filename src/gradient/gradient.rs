use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::Tensor;

pub struct Gradient {
    tensor: Rc<RefCell<Tensor>>,
    gradient: Tensor,
}

impl Gradient {
    pub fn new(tensor: Rc<RefCell<Tensor>>, gradient: Tensor) -> Self {
        debug_assert_eq!(gradient.shape(), tensor.deref().borrow().shape());
        Self { tensor, gradient }
    }
    pub fn tensor(&self) -> &Rc<RefCell<Tensor>> {
        &self.tensor
    }
    pub fn gradient(&self) -> &Tensor {
        &self.gradient
    }
}
