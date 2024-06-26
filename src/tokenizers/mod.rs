mod ascii_tokenizer;
pub use ascii_tokenizer::*;
mod byte_pair_encoding;
pub use byte_pair_encoding::*;

use crate::tensor::Error;

pub trait TokenizerTrait {
    fn vocab_size(&self) -> usize;
    fn encode(&mut self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> Result<String, Error>;
}

pub enum Tokenizer {
    BytePairEncoding(BytePairEncoding),
    AsciiTokenizer(AsciiTokenizer),
}

impl TokenizerTrait for Tokenizer {
    fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::BytePairEncoding(object) => object.vocab_size(),
            Tokenizer::AsciiTokenizer(object) => object.vocab_size(),
        }
    }

    fn encode(&mut self, text: &str) -> Vec<usize> {
        match self {
            Tokenizer::BytePairEncoding(object) => object.encode(text),
            Tokenizer::AsciiTokenizer(object) => object.encode(text),
        }
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, Error> {
        match self {
            Tokenizer::BytePairEncoding(object) => object.decode(tokens),
            Tokenizer::AsciiTokenizer(object) => object.decode(tokens),
        }
    }
}

impl Tokenizer {
    pub fn byte_pair_encoding() -> Tokenizer {
        Tokenizer::BytePairEncoding(BytePairEncoding::default())
    }
    pub fn ascii_tokenizer() -> Tokenizer {
        Tokenizer::AsciiTokenizer(AsciiTokenizer::default())
    }
}
