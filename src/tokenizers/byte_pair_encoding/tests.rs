use std::{collections::HashMap, fs};

use crate::{BytePairEncoding, Tokenizer};

#[test]
fn encode_and_decode() {
    let file_path = "Mega_Man.txt";
    let text = fs::read_to_string(file_path).unwrap();
    let mut tokenizer = BytePairEncoding::default();
    let tokens = tokenizer.encode(&text);
    let decoded_text = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded_text, text);
}

#[ignore]
#[test]
fn no_repeated_pairs() {
    let file_path = "Mega_Man.txt";
    let text = fs::read_to_string(file_path).unwrap();
    let mut tokenizer = BytePairEncoding::default();
    let tokens = tokenizer.encode(&text);
    let mut pairs = HashMap::<(usize, usize), usize>::default();
    for i in 0..tokens.len() - 1 {
        let pair = (tokens[i + 0], tokens[i + 1]);
        pairs
            .entry(pair)
            .and_modify(|counter| *counter += 1)
            .or_insert(1);
    }

    for (_, counter) in pairs.iter() {
        assert_eq!(*counter, 1);
    }
}
