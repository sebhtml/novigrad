use std::fs;

use crate::{
    add_embeddings, get_u8_embedding_table, loss::LossFunctionName, to_multi_class, Activation,
    DatasetDetails, LayerType, LinearConfig, Tensor,
};

fn load_examples() -> Vec<(Tensor, Tensor)> {
    let token_count = 256;
    let context_size = 32;
    let embedding_table = get_u8_embedding_table();
    let mut examples = Vec::new();
    let file_path = "Mega_Man.txt";
    let contents = fs::read_to_string(file_path).expect("contents");
    // TODO use bpe tokenizer.
    let tokens = contents.as_bytes().to_owned();
    println!("[load_megaman_examples] loaded {} tokens", tokens.len());
    let mut i = 0;
    let max_number_of_examples = 10;
    while i + context_size < tokens.len() && i < max_number_of_examples {
        let next_token_index = i + context_size;
        let input_tokens = tokens[i..next_token_index].to_owned();
        let next_token = tokens[next_token_index];
        /*
        println!("input_tokens {:?}", input_tokens);
        println!("next_token {}", next_token);
        */
        let input_embeddings = add_embeddings(&embedding_table, &input_tokens);
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
            LayerType::Linear(LinearConfig {
                input_rows: 32,
                rows: 256,
                cols: 256,
                activation: Activation::Sigmoid,
            }),
            LayerType::Linear(LinearConfig {
                input_rows: 32,
                rows: 256,
                cols: 256,
                activation: Activation::Sigmoid,
            }),
            LayerType::Linear(LinearConfig {
                input_rows: 32,
                rows: 256,
                cols: 256,
                activation: Activation::Softmax,
            }),
        ],
        epochs: 1000,
        progress: 100,
        loss_function_name: LossFunctionName::CrossEntropyLoss,
        initial_total_error_min: 50.0,
        final_total_error_max: 0.002,
    }
}
