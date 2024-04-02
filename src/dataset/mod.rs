mod mega_man;
mod simple;

use crate::{loss::LossFunctionName, LayerType, Tensor};

pub enum Dataset {
    Simple,
    MegaMan,
}

pub struct DatasetDetails {
    pub examples: Vec<(Vec<usize>, Tensor)>,
    pub layers: Vec<LayerType>,
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

pub fn to_multi_class(next_token: usize, token_count: usize) -> Tensor {
    let mut values = vec![];
    values.resize(token_count, 0.0);
    values[next_token as usize] = 1.0;
    Tensor::new(1, token_count, values)
}
