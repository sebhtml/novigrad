use crate::statistics::layer_norm::LayerNormalization;
use crate::tensor::Error;
use crate::transformer::Transformer;
use crate::{Device, Dropout, UnaryModel, UnaryOperator, WeightsInitialization};
use crate::{Embedding, Linear, Model, Softmax, TensorWithGrad};

/// See
/// Full GPT Architecture
/// https://en.wikipedia.org/wiki/GPT-1#/media/File:Full_GPT_architecture.svg
///
/// See
/// Attention Is All You Need
/// https://arxiv.org/pdf/1706.03762
///
/// See
/// OpenAI GPT 1
/// https://huggingface.co/openai-community/openai-gpt
pub struct TransformerModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    dropout: Dropout,
    transformers: Vec<Transformer>,
    layer_norm: LayerNormalization,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for TransformerModel {}

impl TransformerModel {
    pub fn new(
        device: &Device,
        layers: usize,
        sequence_length: usize,
        vocab_size: usize,
    ) -> Result<Self, Error> {
        let n_embd = 768;
        let num_heads = 12;
        let dropout_probability = 0.1;

        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let dropout = Dropout::try_new(device, sequence_length, n_embd, dropout_probability)?;
        let causal_mask = true;
        let transformers = (0..layers)
            .map(|_| {
                Transformer::try_new(
                    device,
                    sequence_length,
                    n_embd,
                    causal_mask,
                    num_heads,
                    dropout_probability,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let layer_norm = LayerNormalization::try_new(device, sequence_length, n_embd)?;
        let linear = Linear::new(
            device,
            vocab_size,
            n_embd,
            WeightsInitialization::Kaiming,
            sequence_length,
        )?;
        let softmax = Softmax::new_with_next_is_cross_entropy_loss(device);

        let model = Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            embedding,
            dropout,
            transformers,
            layer_norm,
            linear,
            softmax,
        };
        Ok(model)
    }
}

impl UnaryOperator for TransformerModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let embedding = self.embedding.forward(input)?;
        let dropout = self.dropout.forward(&embedding)?;
        let mut transformed_outputs = vec![];
        for (layer, transformer) in self.transformers.iter().enumerate() {
            let input = match layer {
                0 => &dropout,
                _ => &transformed_outputs[layer - 1],
            };
            let transformed = transformer.forward(&input)?;
            transformed_outputs.push(transformed);
        }
        let transformed = &transformed_outputs[transformed_outputs.len() - 1];
        let normalized_output = self.layer_norm.forward(&transformed)?;
        let linear = self.linear.forward(&normalized_output)?;
        let softmax = self.softmax.forward(&linear)?;
        Ok(softmax)
    }
}

impl Model for TransformerModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}
