use crate::Tensor;

fn add_simple_embeddings(input: Tensor) -> Tensor {
    let table = vec![
        // 0.0
        vec![0.0], //, 0.0, 1.0, 0.0], //
        // 1.0
        vec![1.0], //, 0.0, 0.0, 0.0], //
    ];
    let mut values = vec![];
    let mut row = 0;
    let rows = input.dimensions()[0];
    while row < rows {
        let index = input.get(&vec![row, 0]) as usize;
        values.append(&mut table[index].clone());
        row += 1;
    }
    Tensor::new(vec![input.dimensions()[0], table[0].len()], values)
}

pub fn load_simple_examples() -> Vec<(Tensor, Tensor)> {
    let mut examples = Vec::new();
    examples.push((
        //
        Tensor::new(vec![4, 1], vec![1.0, 0.0, 0.0, 0.0]), //
        Tensor::new(vec![2, 1], vec![0.1, 0.9]),
    ));
    examples.push((
        //
        Tensor::new(vec![4, 1], vec![1.0, 0.0, 0.0, 1.0]), //
        Tensor::new(vec![2, 1], vec![0.1, 0.9]),
    ));
    examples.push((
        //
        Tensor::new(vec![4, 1], vec![0.0, 0.0, 1.0, 0.0]), //
        Tensor::new(vec![2, 1], vec![0.9, 0.1]),
    ));
    examples.push((
        //
        Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]), //
        Tensor::new(vec![2, 1], vec![0.9, 0.1]),
    ));
    examples
        .into_iter()
        .map(|example| (add_simple_embeddings(example.0), example.1))
        .collect()
}
