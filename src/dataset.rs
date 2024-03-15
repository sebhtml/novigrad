use crate::Matrix;

fn add_simple_embeddings(input: Matrix) -> Matrix {
    let table = vec![
        // 0.0
        vec![0.0], //, 0.0, 1.0, 0.0], //
        // 1.0
        vec![1.0], //, 0.0, 0.0, 0.0], //
    ];
    let mut values = vec![];
    let mut row = 0;
    let rows = input.rows();
    while row < rows {
        let index = input.get(row, 0) as usize;
        values.append(&mut table[index].clone());
        row += 1;
    }
    Matrix::new(input.rows(), table[0].len(), values)
}

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
        .into_iter()
        .map(|example| (add_simple_embeddings(example.0), example.1))
        .collect()
}
