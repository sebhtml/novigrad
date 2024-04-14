use crate::{
    DifferentiableModuleConfig, EmbeddingConfig, LinearConfig, ReshapeConfig, SoftmaxConfig,
};

pub fn architecture() -> Vec<DifferentiableModuleConfig> {
    vec![
        DifferentiableModuleConfig::Embedding(EmbeddingConfig {
            num_embeddings: 16,
            embedding_dim: 32,
        }),
        DifferentiableModuleConfig::Linear(LinearConfig {
            weights_rows: 16,
            weights_cols: 32,
            bias_rows: 6,
        }),
        DifferentiableModuleConfig::Sigmoid(Default::default()),
        DifferentiableModuleConfig::Reshape(ReshapeConfig {
            input_rows: 6,
            input_cols: 16,
            output_rows: 1,
            output_cols: 6 * 16,
        }),
        DifferentiableModuleConfig::Linear(LinearConfig {
            weights_rows: 32,
            weights_cols: 6 * 16,
            bias_rows: 1,
        }),
        DifferentiableModuleConfig::Sigmoid(Default::default()),
        DifferentiableModuleConfig::Linear(LinearConfig {
            weights_rows: 16,
            weights_cols: 32,
            bias_rows: 1,
        }),
        DifferentiableModuleConfig::Softmax(SoftmaxConfig {
            using_softmax_and_cross_entropy_loss: true,
        }),
    ]
}
