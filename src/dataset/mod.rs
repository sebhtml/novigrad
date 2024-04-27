mod mega_man;
mod simple;

use std::rc::Rc;

use crate::{Device, Forward, LearningTensor, Operator};

pub enum Dataset {
    Simple,
    MegaMan,
}

pub struct DatasetDetails {
    pub examples: Vec<(LearningTensor, LearningTensor)>,
    pub architecture: Box<dyn Forward>,
    pub loss_function_name: Operator,
    pub epochs: usize,
    pub progress: usize,
    pub initial_total_error_min: f32,
    pub final_total_error_max: f32,
}

pub fn load_dataset(dataset: Dataset, device: Rc<Device>) -> DatasetDetails {
    match dataset {
        Dataset::Simple => simple::load_dataset(device),
        Dataset::MegaMan => mega_man::load_dataset(device),
    }
}

pub fn into_one_hot_encoded_rows(
    device: &Device,
    input_tokens: &[usize],
    num_classes: usize,
) -> LearningTensor {
    let len = input_tokens.len() * num_classes;
    // TODO avoid allocating a Tensor and a LearningTensor.
    let result = device.tensor(
        input_tokens.len(),
        num_classes,
        vec![Default::default(); len],
    );
    let mut result_values = result.get_values();
    for (index, token) in input_tokens.iter().enumerate() {
        result_values[result.index(index, *token)] = 1.0;
    }
    device.learning_tensor(input_tokens.len(), num_classes, result_values, false)
}
