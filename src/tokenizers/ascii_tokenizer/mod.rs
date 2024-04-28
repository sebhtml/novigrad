use crate::Tokenizer;

#[derive(Default)]
pub struct AsciiTokenizer {}

impl Tokenizer for AsciiTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.as_bytes()
            .to_owned()
            .into_iter()
            .map(|token| token as usize)
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|token| String::from(*token as u8 as char))
            .collect::<Vec<_>>()
            .join("")
    }
}
