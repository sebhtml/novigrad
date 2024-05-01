use std::{collections::HashMap, mem::swap};

use crate::{Error, Tokenizer};

#[cfg(test)]
mod tests;

// TODO remove SHADOW_TOKEN
const SHADOW_TOKEN: usize = 0;
const FIRST_TOKEN: usize = SHADOW_TOKEN + 1;

pub struct BytePairEncoding {
    // TODO add maximum vocabulary size.
    byte_to_token: HashMap<u8, usize>,
    token_to_byte: HashMap<usize, u8>,
    token_pair_to_token: HashMap<(usize, usize), usize>,
    token_to_token_pair: HashMap<usize, (usize, usize)>,
}

impl Default for BytePairEncoding {
    fn default() -> Self {
        Self {
            byte_to_token: Default::default(),
            token_to_byte: Default::default(),
            token_pair_to_token: Default::default(),
            token_to_token_pair: Default::default(),
        }
    }
}

fn get_pair(tokens: &[usize], i: usize, len: usize) -> Option<((usize, usize), usize)> {
    if tokens[i + 0] == SHADOW_TOKEN {
        return None;
    }

    let mut j = 0;

    if i + 1 + j >= len {
        return None;
    }

    while i + 1 + j < len && tokens[i + 1 + j] == SHADOW_TOKEN {
        j += 1;
    }
    let token_1 = tokens[i + 0];
    let token_2 = tokens[i + 1 + j];
    let pair = (token_1, token_2);
    return Some((pair, j));
}

impl Tokenizer for BytePairEncoding {
    fn encode(&mut self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];
        let mut next_token = FIRST_TOKEN;

        let mut allocate_token = || -> usize {
            let token = next_token;
            next_token = next_token + 1;
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
        let mut last_shadow_token_counter = usize::MAX;
        while last_shadow_token_counter > 0 {
            last_shadow_token_counter = 0;

            // Count pairs
            let mut token_pair_counters = HashMap::<(usize, usize), usize>::default();
            let mut i = 0;
            let len = tokens.len();
            while i < len {
                match get_pair(&tokens, i, len) {
                    Some((pair, _)) => {
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
            let pair = max
                .map(|max| token_pair_counters.iter().find(|item| item.1 == max))
                .flatten()
                .map(|item| item.0);
            match pair {
                Some(expected_pair) => {
                    let mut i = 0;
                    while i < len - 1 {
                        match get_pair(&tokens, i, len) {
                            Some((pair, j)) => {
                                //println!("Replacing pair {:?}", pair);
                                if &pair == expected_pair {
                                    let token = allocate_token();
                                    tokens[i + 0] = token;
                                    tokens[i + 1 + j] = SHADOW_TOKEN;
                                    self.token_pair_to_token.insert(pair, token);
                                    self.token_to_token_pair.insert(token, pair);
                                    last_shadow_token_counter += 1;
                                    i += 1 + j + 1;
                                } else {
                                    i += 1;
                                }
                            }
                            _ => {
                                i += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        tokens
            .into_iter()
            .filter(|item| *item != SHADOW_TOKEN)
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, Error> {
        // Decode tokens to pairs.
        let mut tokens2 = tokens.to_owned();
        let mut tokens_tmp = vec![];
        let mut a_token_was_found = true;
        while a_token_was_found {
            tokens_tmp.clear();
            for token in tokens {
                match self.token_to_token_pair.get(token) {
                    Some((token_1, token_2)) => {
                        tokens_tmp.push(*token_1);
                        tokens_tmp.push(*token_2);
                        a_token_was_found = true;
                    }
                    _ => {
                        tokens_tmp.push(*token);
                        a_token_was_found = false;
                    }
                }
            }
            swap(&mut tokens2, &mut tokens_tmp);
        }

        // Decode tokens to bytes
        let mut output = vec![];
        for token in tokens2 {
            let byte = self
                .token_to_byte
                .get(&token)
                .ok_or(Error::UnsupportedOperation)?;
            output.push(*byte);
        }
        String::from_utf8(output).map_err(|_| Error::UnsupportedOperation)
    }
}
