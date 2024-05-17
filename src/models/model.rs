use crate::UnaryOperator;

pub trait Model {
    fn input_size(&self) -> Vec<usize>;
    fn output_size(&self) -> Vec<usize>;
}

pub trait UnaryModel: UnaryOperator + Model {}
