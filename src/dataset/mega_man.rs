use std::fs;

use crate::{
    loss::LossFunctionType, to_multi_class, ActivationType, DatasetDetails, EmbeddingConfig,
    LayerConfig, LayerType, LinearConfig, ReshapeConfig, Tensor,
};

fn load_examples() -> Vec<(Tensor, Tensor)> {
    let token_count = 256;
    let context_size = 32;
    let mut examples = Vec::new();
    let file_path = "Mega_Man.txt";
    let contents = fs::read_to_string(file_path).expect("contents");
    // TODO use bpe tokenizer.
    let tokens: Vec<usize> = contents
        .as_bytes()
        .to_owned()
        .into_iter()
        .map(|token| token as usize)
        .collect();
    println!("[load_megaman_examples] loaded {} tokens", tokens.len());
    let mut i = 0;
    let max_number_of_examples = 10;
    while i + context_size < tokens.len() && i < max_number_of_examples {
        let next_token_index = i + context_size;
        let input_tokens = tokens[i..next_token_index].to_owned();
        let next_token = tokens[next_token_index];

        let output_multiclass = to_multi_class(next_token, token_count);

        examples.push((
            //
            input_tokens.into(), //
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
            LayerConfig::Embedding(EmbeddingConfig {
                hidden_dimensions: 256,
            }),
            LayerConfig::Reshape(ReshapeConfig {
                input_rows: 32,
                input_cols: 256,
                output_rows: 1,
                output_cols: 32 * 256,
            }),
            LayerConfig::Linear(LinearConfig {
                input_rows: 1,
                rows: 256,
                cols: 32 * 256,
                activation: ActivationType::Softmax(Default::default()),
            }),
        ],
        epochs: 300,
        progress: 100,
        loss_function_name: LossFunctionType::CrossEntropyLoss(Default::default()),
        initial_total_error_min: 50.0,
        final_total_error_max: 0.002,
    }
}
