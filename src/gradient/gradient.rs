use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{Accelerator, Tensor};

pub struct Gradient {
    tensor: Rc<RefCell<Tensor>>,
    gradient: Tensor,
}

impl Gradient {
    pub fn new(tensor: Rc<RefCell<Tensor>>, gradient: Tensor) -> Self {
        Self { tensor, gradient }
    }
    pub fn commit_change(&self, accelerator: &Accelerator, learning_rate: f32) {
        let tensor: &mut Tensor = &mut self.tensor.deref().borrow_mut();
        let op_result = Tensor::saxpy(accelerator, -learning_rate, &self.gradient, tensor);
        op_result.expect("Ok");
    }
}
