mod mega_man;
mod simple;

use crate::{loss::LossFunctionType, DifferentiableModuleConfig, Tensor};

pub enum Dataset {
    Simple,
    MegaMan,
}

pub struct DatasetDetails {
    pub examples: Vec<(Tensor, Tensor)>,
    pub layers: Vec<DifferentiableModuleConfig>,
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

pub fn into_one_hot_encoded_rows(input_tokens: &[usize], num_classes: usize, result: &mut Tensor) {
    result.reset(input_tokens.len(), num_classes, Default::default());
    for (index, token) in input_tokens.iter().enumerate() {
        result.set(index, *token, 1.0);
    }
}
