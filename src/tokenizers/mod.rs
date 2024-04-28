mod ascii_tokenizer;
pub use ascii_tokenizer::*;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
}
