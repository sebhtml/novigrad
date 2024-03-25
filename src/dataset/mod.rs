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
}

pub fn load_dataset(dataset: &Dataset) -> DatasetDetails {
    match dataset {
        Dataset::Simple => simple::load_dataset(),
        Dataset::MegaMan => mega_man::load_dataset(),
    }
}
