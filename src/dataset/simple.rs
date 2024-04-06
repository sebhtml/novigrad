use crate::{
    loss::LossFunctionType, ActivationType, DatasetDetails, EmbeddingConfig, LayerConfig,
    LayerType, LinearConfig, ReshapeConfig, Tensor,
};

fn load_examples() -> Vec<(Tensor, Tensor)> {
    let mut examples = Vec::new();

    examples.push((
        //
        vec![1, 2, 3, 4, 5, 6], //
        vec![1.0, 0.0, 0.0, 0.0],
    ));

    examples.push((
        //
        vec![7, 8, 9, 10, 11, 12], //
        vec![0.0, 0.0, 0.0, 1.0],
    ));

    let examples = examples
        .into_iter()
        .map(|example| (example.0.into(), Tensor::new(1, example.1.len(), example.1)))
        .collect();

    examples
}

pub fn load_dataset() -> DatasetDetails {
    DatasetDetails {
        examples: load_examples(),
        layers: vec![
            LayerConfig::Embedding(EmbeddingConfig {
                hidden_dimensions: 256,
            }),
            LayerConfig::Linear(LinearConfig {
                input_rows: 6,
                rows: 256,
                cols: 256,
                activation: ActivationType::Sigmoid(Default::default()),
            }),
            LayerConfig::Reshape(ReshapeConfig {
                input_rows: 6,
                input_cols: 256,
                output_rows: 1,
                output_cols: 6 * 256,
            }),
            LayerConfig::Linear(LinearConfig {
                input_rows: 1,
                rows: 256,
                cols: 6 * 256,
                activation: ActivationType::Sigmoid(Default::default()),
            }),
            LayerConfig::Linear(LinearConfig {
                input_rows: 1,
                rows: 4,
                cols: 256,
                activation: ActivationType::Softmax(Default::default()),
            }),
        ],
        epochs: 1000,
        progress: 100,
        loss_function_name: LossFunctionType::CrossEntropyLoss(Default::default()),
        initial_total_error_min: 2.0,
        final_total_error_max: 0.00025,
    }
}
