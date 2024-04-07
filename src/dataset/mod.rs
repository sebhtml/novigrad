mod mega_man;
mod simple;

use crate::{loss::LossFunctionType, LayerConfig, Tensor, TensorTrait};

pub enum Dataset {
    Simple,
    MegaMan,
}

pub struct DatasetDetails {
    pub examples: Vec<(Tensor, Tensor)>,
    pub layers: Vec<LayerConfig>,
    pub loss_function_name: LossFunctionType,
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

// TODO remove to_multi_class and use into_one_hot_encoded_rows instead.
pub fn to_multi_class(next_token: usize, token_count: usize) -> Tensor {
    let mut values = vec![];
    values.resize(token_count, 0.0);
    values[next_token as usize] = 1.0;
    Tensor::new(1, token_count, values)
}

pub fn into_one_hot_encoded_rows(input_tokens: &[usize], num_classes: usize, result: &mut Tensor) {
    result.reset(input_tokens.len(), num_classes, Default::default());
    for (index, token) in input_tokens.iter().enumerate() {
        result.set(index, *token, 1.0);
    }
}
