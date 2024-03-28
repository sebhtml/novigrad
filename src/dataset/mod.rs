mod mega_man;
mod simple;
use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{loss::LossFunctionName, LayerConfig, Tensor};

pub enum Dataset {
    Simple,
    MegaMan,
}

pub struct DatasetDetails {
    pub examples: Vec<(Tensor, Tensor)>,
    pub layers: Vec<LayerConfig>,
    pub loss_function_name: LossFunctionName,
    pub epochs: usize,
    pub progress: usize,
    pub initial_total_error_min: f32,
    pub final_total_error_max: f32,
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

pub fn add_embeddings(embedding_table: &Vec<Vec<f32>>, input: &Vec<u8>) -> Tensor {
    let mut values = vec![];
    let mut row = 0;
    let rows = input.len();
    while row < rows {
        let index = input[row];
        values.append(&mut embedding_table[index as usize].clone());
        row += 1;
    }
    Tensor::new(input.len(), embedding_table[0].len(), values)
}
