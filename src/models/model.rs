pub trait Model {
    fn input_shape(&self) -> Vec<usize>;
    fn output_shape(&self) -> Vec<usize>;
}
