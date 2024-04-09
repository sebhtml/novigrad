use std::mem::swap;

use crate::Tensor;

pub struct DifferentiableTensor {
    pub tensor: Tensor,
    pub gradient: Tensor,
    pub has_gradient: bool,
    tmp: Tensor,
    addition: Tensor,
}

impl DifferentiableTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            gradient: Default::default(),
            has_gradient: Default::default(),
            tmp: Default::default(),
            addition: Default::default(),
        }
    }
    pub fn commit_change(&mut self, learning_rate: f32) {
        if !self.has_gradient {
            return;
        }
        let tmp = &mut self.tmp;
        let addition = &mut self.addition;
        let op_result = self.gradient.scalar_mul(-learning_rate, tmp);
        op_result.expect("Ok");
        let op_result = self.tensor.add(&tmp, addition);
        op_result.expect("Ok");
        swap(&mut self.tensor, addition);
        self.has_gradient = false;
    }
}

impl From<Tensor> for DifferentiableTensor {
    fn from(value: Tensor) -> Self {
        Self::new(value)
    }
}
