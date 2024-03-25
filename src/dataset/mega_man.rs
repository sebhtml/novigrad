use std::fs;

use crate::{Activation, DatasetDetails, LayerConfig, Tensor};

fn add_embeddings(
    embeddings_table: &Vec<Vec<f32>>,
    num_embeddings: usize,
    embedding_dim: usize,
    input: Vec<u8>,
) -> Tensor {
    let mut values = vec![];
    let mut col = 0;
    while col < num_embeddings {
        let token = input[col];
        values.append(&mut embeddings_table[token as usize].clone());
        col += 1;
    }
    Tensor::new(1, num_embeddings * embedding_dim, values)
}

fn to_multi_class(next_token: u8, token_count: usize) -> Tensor {
    let mut values = vec![];
    values.resize(token_count, 0.0);
    values[next_token as usize] = 1.0;
    Tensor::new(1, token_count, values)
}

pub fn get_u8_embedding_table() -> Vec<Vec<f32>> {
    let mut embeddings_table = Vec::new();
    let mut token = 0;
    while token < 256 {
        let token_embeddings: Vec<f32> = vec![
            (token >> 0) & 0x01,
            (token >> 1) & 0x01,
            (token >> 2) & 0x01,
            (token >> 3) & 0x01,
            (token >> 4) & 0x01,
            (token >> 5) & 0x01,
            (token >> 6) & 0x01,
            (token >> 7) & 0x01,
        ]
        .into_iter()
        .map(|x| x as f32)
        .collect();
        embeddings_table.push(token_embeddings);
        token += 1;
    }
    embeddings_table
}

fn load_examples() -> Vec<(Tensor, Tensor)> {
    let token_count = 256;
    let context_size = 32;
    let embedding_dim = 8;
    let embedding_table = get_u8_embedding_table();
    let mut examples = Vec::new();
    let file_path = "Mega_Man.txt";
    let contents = fs::read_to_string(file_path).expect("contents");
    // TODO use bpe tokenizer.
    let tokens = contents.as_bytes().to_owned();
    println!("[load_megaman_examples] loaded {} tokens", tokens.len());
    let mut i = 0;
    while i + context_size < tokens.len() && i < 100 {
        let next_token_index = i + context_size;
        let num_embeddings = context_size;
        let input_tokens = tokens[i..next_token_index].to_owned();
        let next_token = tokens[next_token_index];
        /*
        println!("input_tokens {:?}", input_tokens);
        println!("next_token {}", next_token);
         */
        let input_embeddings = add_embeddings(
            &embedding_table,
            num_embeddings,
            embedding_dim,
            input_tokens,
        );
        let output_multiclass = to_multi_class(next_token, token_count);

        examples.push((
            //
            input_embeddings, //
            output_multiclass,
        ));
        i += 1;
    }
    examples
}

pub fn load_dataset() -> DatasetDetails {
    DatasetDetails {
        examples: load_examples(),
        layers: vec![
            LayerConfig {
                rows: 4,
                cols: 8,
                activation: Activation::Sigmoid,
            },
            LayerConfig {
                rows: 8,
                cols: 4,
                activation: Activation::Sigmoid,
            },
            LayerConfig {
                rows: 4,
                cols: 8,
                activation: Activation::Softmax,
            },
        ],
        epochs: 1000000,
        progress: 10000,
    }
}
