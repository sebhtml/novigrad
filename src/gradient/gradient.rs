use crate::{Accelerator, Tensor};

pub struct Gradient {
    pub tensor: Tensor,
    pub gradient: Tensor,
    pub has_gradient: bool,
}

impl Gradient {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            gradient: Default::default(),
            has_gradient: Default::default(),
        }
    }
    pub fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32) {
        if !self.has_gradient {
            return;
        }

        let op_result = Tensor::saxpy(
            accelerator,
            -learning_rate,
            &self.gradient,
            &mut self.tensor,
        );
        op_result.expect("Ok");
        self.has_gradient = false;
    }
}

impl From<Tensor> for Gradient {
    fn from(value: Tensor) -> Self {
        Self::new(value)
    }
}
