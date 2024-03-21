use crate::Tensor;

// num_embeddings = 2
// embedding_dim = 4
fn get_embedding_table() -> Vec<Vec<f32>> {
    let embeddings_table = vec![
        // 0.0
        vec![0.0, 0.0, 1.0, 0.0], //
        // 1.0
        vec![1.0, 0.0, 0.0, 0.0], //
    ];
    embeddings_table
}

// TODO the input should be a Tensor with integers, not a Tensor with floats
fn add_simple_embeddings(embedding_table: &Vec<Vec<f32>>, input: &Tensor) -> Tensor {
    let mut values = vec![];
    let mut row = 0;
    let rows = input.rows();
    while row < rows {
        let index = input.get(row, 0) as usize;
        values.append(&mut embedding_table[index].clone());
        row += 1;
    }
    Tensor::new(input.rows(), embedding_table[0].len(), values)
}

pub fn load_simple_examples() -> Vec<(Tensor, Tensor)> {
    let embedding_table = get_embedding_table();
    let mut examples = Vec::new();
    examples.push((
        //
        Tensor::new(6, 1, vec![0.9, 0.1, 0.1, 0.1, 0.1, 0.1]), //
        Tensor::new(1, 1, vec![0.1]),
    ));
    examples.push((
        //
        Tensor::new(6, 1, vec![0.9, 0.1, 0.1, 0.1, 0.1, 0.1]), //
        Tensor::new(1, 1, vec![0.1]),
    ));
    examples.push((
        //
        Tensor::new(6, 1, vec![0.1, 0.1, 0.9, 0.1, 0.1, 0.1]), //
        Tensor::new(1, 1, vec![0.9]),
    ));
    examples.push((
        //
        Tensor::new(6, 1, vec![0.1, 0.1, 0.9, 0.1, 0.1, 0.1]), //
        Tensor::new(1, 1, vec![0.9]),
    ));

    // TODO instead of manually adding constant embeddings, they should be learned.
    examples
    /*
    .into_iter()
    .map(|example| {
        (
            add_simple_embeddings(&embedding_table, &example.0),
            example.1,
        )
    })
    .collect()
     */
}
