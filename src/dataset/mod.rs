mod mega_man;
mod simple;
use rand::{distributions::Uniform, thread_rng, Rng};

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
    let mut rng = thread_rng();
    let mut embeddings_table = Vec::new();
    let mut token = 0;
    let left = 0.0;
    let right = 1.0;
    let width = 256;
    let uniform = Uniform::new(left, right);
    while token < width {
        let mut token_embeddings: Vec<f32> = Vec::new();
        for _ in 0..width {
            let value = rng.sample(uniform);
            token_embeddings.push(value);
        }
        embeddings_table.push(token_embeddings);
        token += 1;
    }
    embeddings_table
}
