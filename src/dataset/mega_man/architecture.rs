use crate::{
    DifferentiableModuleConfig, EmbeddingConfig, LinearConfig, ReshapeConfig, SoftmaxConfig,
};

pub fn architecture() -> Vec<DifferentiableModuleConfig> {
    vec![
        DifferentiableModuleConfig::Embedding(EmbeddingConfig {
            num_embeddings: 256,
            embedding_dim: 384,
        }),
        DifferentiableModuleConfig::Reshape(ReshapeConfig {
            input_rows: 32,
            input_cols: 384,
            output_rows: 1,
            output_cols: 32 * 384,
        }),
        DifferentiableModuleConfig::Linear(LinearConfig {
            weights_rows: 256,
            weights_cols: 32 * 384,
            bias_rows: 1,
        }),
        DifferentiableModuleConfig::Softmax(SoftmaxConfig {
            using_softmax_and_cross_entropy_loss: true,
        }),
    ]
}
