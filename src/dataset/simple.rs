use crate::{
    into_one_hot_encoded_rows, loss::LossFunctionType, DatasetDetails, EmbeddingConfig,
    LayerConfig, LinearConfig, ReshapeConfig, SoftmaxConfig, Tensor,
};

fn load_examples() -> Vec<(Tensor, Tensor)> {
    let mut examples = Vec::new();

    examples.push((
        //
        vec![1, 2, 3, 4, 5, 6], //
        vec![0],
    ));

    examples.push((
        //
        vec![7, 8, 9, 10, 11, 12], //
        vec![3],
    ));

    let num_classes = 16;
    let mut one_hot_encoded_input = Tensor::default();
    let mut one_hot_encoded_output = Tensor::default();
    let examples = examples
        .into_iter()
        .map(|example| {
            into_one_hot_encoded_rows(&example.0, num_classes, &mut one_hot_encoded_input);
            into_one_hot_encoded_rows(&example.1, num_classes, &mut one_hot_encoded_output);
            (
                one_hot_encoded_input.clone(),
                one_hot_encoded_output.clone(),
            )
        })
        .collect();

    examples
}

pub fn load_dataset() -> DatasetDetails {
    DatasetDetails {
        examples: load_examples(),
        layers: vec![
            LayerConfig::Embedding(EmbeddingConfig {
                num_embeddings: 16,
                embedding_dim: 32,
            }),
            LayerConfig::Linear(LinearConfig {
                input_rows: 6,
                rows: 16,
                cols: 32,
            }),
            LayerConfig::Sigmoid(Default::default()),
            LayerConfig::Reshape(ReshapeConfig {
                input_rows: 6,
                input_cols: 16,
                output_rows: 1,
                output_cols: 6 * 16,
            }),
            LayerConfig::Linear(LinearConfig {
                input_rows: 1,
                rows: 32,
                cols: 6 * 16,
            }),
            LayerConfig::Sigmoid(Default::default()),
            LayerConfig::Linear(LinearConfig {
                input_rows: 1,
                rows: 16,
                cols: 32,
            }),
            LayerConfig::Softmax(SoftmaxConfig {
                using_softmax_and_cross_entropy_loss: true,
            }),
        ],
        epochs: 1000,
        progress: 100,
        loss_function_name: LossFunctionType::CrossEntropyLoss(Default::default()),
        initial_total_error_min: 4.0,
        final_total_error_max: 0.0004,
    }
}
