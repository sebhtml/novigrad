mod simple;
pub use simple::*;

mod mega_man;
pub use mega_man::*;

use crate::Tensor;

pub enum Dataset {
    Simple,
    MegaMan,
}

pub fn load_examples(dataset: Dataset) -> Vec<(Tensor, Tensor)> {
    match dataset {
        Dataset::Simple => load_simple_examples(),
        Dataset::MegaMan => load_megaman_examples(),
    }
}