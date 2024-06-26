use crate::{tensor::Error, TokenizerTrait};

#[derive(Default)]
pub struct AsciiTokenizer {}

impl TokenizerTrait for AsciiTokenizer {
    fn encode(&mut self, text: &str) -> Vec<usize> {
        text.as_bytes()
            .to_owned()
            .into_iter()
            .map(|token| token as usize)
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, Error> {
        let output = tokens
            .iter()
            .map(|token| String::from(*token as u8 as char))
            .collect::<Vec<_>>()
            .join("");
        Ok(output)
    }

    fn vocab_size(&self) -> usize {
        256
    }
}
