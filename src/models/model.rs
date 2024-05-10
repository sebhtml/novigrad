use crate::{Error, Tensor};

pub trait Model {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error>;
    fn input_shape(&self) -> Vec<usize>;
    fn output_shape(&self) -> Vec<usize>;
}
