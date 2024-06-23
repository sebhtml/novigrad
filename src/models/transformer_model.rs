use super::load_examples;
use crate::transformer::Transformer;
use crate::{tensor::Error, ModelDetails};
use crate::{
    Adam, Device, Dropout, Metrics, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait, UnaryModel,
    UnaryOperator, WeightsInitialization,
};
use crate::{Embedding, Linear, Model, Softmax, TensorWithGrad};

pub struct TransformerModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    dropout: Dropout,
    transformer: Transformer,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for TransformerModel {}

impl TransformerModel {
    pub fn new(device: &Device, sequence_length: usize, vocab_size: usize) -> Result<Self, Error> {
        let n_embd = 768;
        let num_heads = 12;
        let dropout_probability = 0.1;

        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let dropout = Dropout::try_new(device, sequence_length, n_embd, dropout_probability)?;
        let causal_mask = true;
        let transformer = Transformer::try_new(
            device,
            sequence_length,
            n_embd,
            causal_mask,
            num_heads,
            dropout_probability,
        )
        .unwrap();
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
            transformer,
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
        let transformed = self.transformer.forward(&dropout)?;
        let linear = self.linear.forward(&transformed)?;
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

pub fn load_transformer_model(
    device: &Device,
) -> Result<ModelDetails<TransformerModel, SoftmaxCrossEntropyLoss, Adam>, Error> {
    let file_path = "data/Geoffrey_Hinton.txt";
    let max_chars = None;
    let max_number_of_examples = 16;
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 64;

    let input_sequence_length = sequence_length;
    let output_sequence_length = sequence_length;
    let examples = load_examples(
        device,
        file_path,
        max_chars,
        max_number_of_examples,
        input_sequence_length,
        output_sequence_length,
        &mut tokenizer,
    )?;

    let vocab_size = tokenizer.vocab_size();
    let model = TransformerModel::new(device, sequence_length, vocab_size)?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let details = ModelDetails {
        device: device.clone(),
        tokenizer: Some(tokenizer),
        examples,
        model,
        loss_operator,
        optimizer,
        epochs: 200,
        progress: 10,
        learning_rate,
        shuffle_examples: true,
        clipped_gradient_norm: false,
        initial_metrics: Metrics {
            total_loss: 4000.0,
            total_perplexity: 5.0,
        },
        final_metrics: Metrics {
            total_loss: 350.0,
            total_perplexity: 20.0,
        },
    };
    Ok(details)
}
