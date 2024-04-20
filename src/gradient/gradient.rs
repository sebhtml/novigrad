use crate::{Accelerator, Tensor};

pub struct Gradient {
    pub tensor: Tensor,
    pub gradient: Tensor,
}

impl Gradient {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            gradient: Default::default(),
        }
    }
    pub fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32) {
        let op_result = Tensor::saxpy(
            accelerator,
            -learning_rate,
            &self.gradient,
            &mut self.tensor,
        );
        op_result.expect("Ok");
    }
}

impl From<Tensor> for Gradient {
    fn from(value: Tensor) -> Self {
        Self::new(value)
    }
}
