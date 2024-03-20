use crate::Tensor;

fn add_simple_embeddings(input: Tensor) -> Tensor {
    let embeddings_table = vec![
        // 0.0
        vec![0.0, 0.0, 1.0, 0.0], //
        // 1.0
        vec![1.0, 0.0, 0.0, 0.0], //
    ];
    let mut values = vec![];
    let mut col = 0;
    let cols = input.cols();
    while col < cols {
        let index = input.get(0, col) as usize;
        values.append(&mut embeddings_table[index].clone());
        col += 1;
    }
    Tensor::new(1, input.rows() * embeddings_table[0].len(), values)
}

pub fn load_simple_examples() -> Vec<(Tensor, Tensor)> {
    let mut examples = Vec::new();
    examples.push((
        //
        Tensor::new(1, 4, vec![1.0, 0.0, 0.0, 0.0]), //
        Tensor::new(1, 2, vec![0.1, 0.9]),
    ));
    examples.push((
        //
        Tensor::new(1, 4, vec![1.0, 0.0, 0.0, 1.0]), //
        Tensor::new(1, 2, vec![0.1, 0.9]),
    ));
    examples.push((
        //
        Tensor::new(1, 4, vec![0.0, 0.0, 1.0, 0.0]), //
        Tensor::new(1, 2, vec![0.9, 0.1]),
    ));
    examples.push((
        //
        Tensor::new(1, 4, vec![0.0, 1.0, 1.0, 0.0]), //
        Tensor::new(1, 2, vec![0.9, 0.1]),
    ));
    examples
        .into_iter()
        .map(|example| (add_simple_embeddings(example.0), example.1))
        .collect()
}
