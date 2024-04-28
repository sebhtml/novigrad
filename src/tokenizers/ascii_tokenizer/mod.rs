use crate::Tokenizer;

#[derive(Default)]
pub struct AsciiTokenizer {}

impl Tokenizer for AsciiTokenizer {
    fn encode(&self, text: &String) -> Vec<usize> {
        text.as_bytes()
            .to_owned()
            .into_iter()
            .map(|token| token as usize)
            .collect()
    }

    fn decode(&self, _tokens: &[usize]) -> String {
        "".into()
    }
}
