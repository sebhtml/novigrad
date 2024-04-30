use std::collections::HashMap;

use crate::{Error, Tokenizer};

#[cfg(test)]
mod tests;

pub struct BytePairEncoding {
    u8_to_usize: HashMap<u8, usize>,
    usize_to_u8: HashMap<usize, u8>,
    _pair_to_usize: HashMap<(usize, usize), usize>,
}

impl Default for BytePairEncoding {
    fn default() -> Self {
        Self {
            u8_to_usize: Default::default(),
            usize_to_u8: Default::default(),
            _pair_to_usize: Default::default(),
        }
    }
}

impl Tokenizer for BytePairEncoding {
    fn encode(&mut self, text: &str) -> Vec<usize> {
        let mut output = vec![];
        let mut next_token = 0;
        for byte in text.bytes() {
            let token = self.u8_to_usize.entry(byte).or_insert_with(|| {
                let token = next_token;
                next_token = next_token + 1;
                self.usize_to_u8.insert(token, byte);
                token
            });
            output.push(*token);
        }
        output
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, Error> {
        let mut output = vec![];
        for token in tokens {
            let byte = self
                .usize_to_u8
                .get(token)
                .ok_or(Error::UnsupportedOperation)?;
            output.push(*byte);
        }
        String::from_utf8(output).map_err(|_| Error::UnsupportedOperation)
    }
}
