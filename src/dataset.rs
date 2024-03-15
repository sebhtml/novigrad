use crate::Matrix;

pub fn load_simple_examples() -> Vec<(Matrix, Matrix)> {
    let mut examples = Vec::new();
    examples.push((
        //
        Matrix::new(4, 1, vec![1.0, 0.0, 0.0, 0.0]), //
        Matrix::new(2, 1, vec![0.1, 0.9]),
    ));
    examples.push((
        //
        Matrix::new(4, 1, vec![1.0, 0.0, 0.0, 1.0]), //
        Matrix::new(2, 1, vec![0.1, 0.9]),
    ));
    examples.push((
        //
        Matrix::new(4, 1, vec![0.0, 0.0, 1.0, 0.0]), //
        Matrix::new(2, 1, vec![0.9, 0.1]),
    ));
    examples.push((
        //
        Matrix::new(4, 1, vec![0.0, 1.0, 1.0, 0.0]), //
        Matrix::new(2, 1, vec![0.9, 0.1]),
    ));
    examples
}
