use std::{collections::BTreeMap, collections::HashMap, mem::swap};

use crate::{error, tensor::Error, tensor::ErrorEnum, TokenizerTrait};

#[cfg(test)]
mod tests;

#[derive(Default)]
pub struct BytePairEncoding {
    vocab_size: usize,
    // TODO add maximum vocabulary size.
    byte_to_token: HashMap<u8, usize>,
    token_to_byte: HashMap<usize, u8>,
    token_pair_to_token: HashMap<(usize, usize), usize>,
    token_to_token_pair: HashMap<usize, (usize, usize)>,
}

fn get_pair(tokens: &[usize], i: usize) -> Option<(usize, usize)> {
    if i + 1 >= tokens.len() {
        return None;
    }

    let token_1 = tokens[i];
    let token_2 = tokens[i + 1];
    let pair = (token_1, token_2);
    Some(pair)
}

impl TokenizerTrait for BytePairEncoding {
    fn encode(&mut self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];
        let mut tokens_tmp = vec![];
        let mut next_token = 0;

        let mut allocate_token = || -> usize {
            let token = next_token;
            next_token += 1;
            token
        };

        // Encode bytes into tokens
        for byte in text.bytes() {
            let token = self.byte_to_token.entry(byte).or_insert_with(|| {
                let token = allocate_token();
                self.token_to_byte.insert(token, byte);
                token
            });
            tokens.push(*token);
        }

        // Encode token pairs into tokens
        let mut token_pair_counters = BTreeMap::<(usize, usize), usize>::default();
        let mut has_repeated_pair = true;
        while has_repeated_pair {
            // Count pairs
            token_pair_counters.clear();
            let mut i = 0;
            while i < tokens.len() {
                match get_pair(&tokens, i) {
                    Some(pair) => {
                        token_pair_counters
                            .entry(pair)
                            .and_modify(|counter| *counter += 1)
                            .or_insert(1);
                    }
                    _ => {}
                }
                i += 1;
            }
            let max = token_pair_counters
                .iter()
                .filter(|item| item.1 > &1)
                .map(|item| item.1)
                .max();

            let expected_pair = max
                .and_then(|max| token_pair_counters.iter().find(|item| item.1 == max))
                .map(|item| item.0);

            has_repeated_pair = expected_pair.is_some();
            match expected_pair {
                Some(expected_pair) => {
                    tokens_tmp.clear();
                    let mut i = 0;
                    while i < tokens.len() {
                        match get_pair(&tokens, i) {
                            Some(pair) => {
                                if &pair == expected_pair {
                                    let token = allocate_token();
                                    tokens_tmp.push(token);
                                    self.token_pair_to_token.insert(pair, token);
                                    self.token_to_token_pair.insert(token, pair);
                                    i += 1 + 1;
                                } else {
                                    tokens_tmp.push(tokens[i]);
                                    i += 1;
                                }
                            }
                            _ => {
                                tokens_tmp.push(tokens[i]);
                                i += 1;
                            }
                        }
                    }
                    swap(&mut tokens, &mut tokens_tmp);
                }
                _ => {}
            }
        }

        self.vocab_size = next_token;

        tokens
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, Error> {
        // Decode tokens to pairs.
        let mut tokens2 = tokens.to_owned();
        let mut tokens_tmp = vec![];
        let mut a_token_was_found = true;
        while a_token_was_found {
            tokens_tmp.clear();
            a_token_was_found = false;
            for token in tokens2.iter() {
                match self.token_to_token_pair.get(token) {
                    Some((token_1, token_2)) => {
                        tokens_tmp.push(*token_1);
                        tokens_tmp.push(*token_2);
                        a_token_was_found = true;
                    }
                    _ => {
                        tokens_tmp.push(*token);
                    }
                }
            }
            swap(&mut tokens2, &mut tokens_tmp);
        }

        // Decode tokens to bytes
        let mut output = vec![];
        for token in tokens2 {
            let byte = self.token_to_byte.get(&token).unwrap_or(&b'?');
            output.push(*byte);
        }
        String::from_utf8(output).map_err(|_| error!(ErrorEnum::UnsupportedOperation))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
