mod mega_man;
mod simple;
use crate::{LayerConfig, Tensor};

pub enum Dataset {
    Simple,
    MegaMan,
}

pub struct DatasetDetails {
    pub examples: Vec<(Tensor, Tensor)>,
    pub layers: Vec<LayerConfig>,
    pub epochs: usize,
    pub progress: usize,
}

pub fn load_dataset(dataset: &Dataset) -> DatasetDetails {
    match dataset {
        Dataset::Simple => simple::load_dataset(),
        Dataset::MegaMan => mega_man::load_dataset(),
    }
}

pub fn to_multi_class(next_token: u8, token_count: usize) -> Tensor {
    let mut values = vec![];
    values.resize(token_count, 0.0);
    values[next_token as usize] = 1.0;
    Tensor::new(1, token_count, values)
}

pub fn get_u8_embedding_table() -> Vec<Vec<f32>> {
    let mut embeddings_table = Vec::new();
    let mut token = 0;
    while token < 256 {
        let token_embeddings: Vec<f32> = vec![
            (token >> 0) & 0x01,
            (token >> 1) & 0x01,
            (token >> 2) & 0x01,
            (token >> 3) & 0x01,
            (token >> 4) & 0x01,
            (token >> 5) & 0x01,
            (token >> 6) & 0x01,
            (token >> 7) & 0x01,
        ]
        .into_iter()
        .map(|x| x as f32)
        .collect();
        embeddings_table.push(token_embeddings);
        token += 1;
    }
    embeddings_table
}
