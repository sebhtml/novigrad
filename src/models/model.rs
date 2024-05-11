pub trait Model {
    fn input_size(&self) -> Vec<usize>;
    fn output_size(&self) -> Vec<usize>;
}
