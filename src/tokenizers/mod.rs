mod ascii_tokenizer;
pub use ascii_tokenizer::*;
mod byte_pair_encoding;
pub use byte_pair_encoding::*;

use crate::Error;

pub trait Tokenizer {
    fn encode(&mut self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> Result<String, Error>;
}
