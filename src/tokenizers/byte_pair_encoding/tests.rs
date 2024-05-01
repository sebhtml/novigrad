use std::{collections::HashMap, fs};

use crate::{BytePairEncoding, Tokenizer};

#[test]
fn decode() {
    let file_path = "Mega_Man.txt";
    let text = fs::read_to_string(file_path).unwrap();
    let mut tokenizer = BytePairEncoding::default();
    let tokens = tokenizer.encode(&text);
    let decoded_text = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded_text, text);
}

#[test]
fn text_length() {
    let file_path = "Mega_Man.txt";
    let text = fs::read_to_string(file_path).unwrap();
    assert_eq!(text.len(), 78970,);
}

#[test]
fn deterministic_tokens_length() {
    let file_path = "Mega_Man.txt";
    let text = fs::read_to_string(file_path).unwrap();
    let mut tokenizer = BytePairEncoding::default();
    let tokens = tokenizer.encode(&text);
    assert_eq!(tokens.len(), 44513,);
}

#[test]
fn no_repeated_pairs() {
    let file_path = "Mega_Man.txt";
    let text = fs::read_to_string(file_path).unwrap();
    let mut tokenizer = BytePairEncoding::default();
    let tokens = tokenizer.encode(&text);
    let mut token_pair_counters = HashMap::<(usize, usize), usize>::default();
    for i in 0..tokens.len() - 1 {
        let pair = (tokens[i + 0], tokens[i + 1]);
        token_pair_counters
            .entry(pair)
            .and_modify(|counter| *counter += 1)
            .or_insert(1);
    }

    for (_, counter) in token_pair_counters.iter() {
        assert_eq!(*counter, 1);
    }
}
