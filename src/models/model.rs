use std::ops::Deref;

use crate::UnaryOperator;

pub trait Model {
    fn input_size(&self) -> Vec<usize>;
    fn output_size(&self) -> Vec<usize>;
}

pub trait UnaryModel: UnaryOperator + Model {}

impl Model for Box<dyn UnaryModel> {
    fn input_size(&self) -> Vec<usize> {
        self.deref().input_size()
    }

    fn output_size(&self) -> Vec<usize> {
        self.deref().output_size()
    }
}

impl UnaryOperator for Box<dyn UnaryModel> {
    fn forward(&self, input: &crate::Tensor) -> Result<crate::Tensor, crate::Error> {
        self.deref().forward(input)
    }
}

impl UnaryModel for Box<dyn UnaryModel> {}
