use super::load_examples;
use crate::{
    Adam, Device, Metrics, MultiHeadAttention, SoftmaxCrossEntropyLoss, TernaryOperator, Tokenizer,
    TokenizerTrait, UnaryModel, UnaryOperator,
};
use crate::{Embedding, Linear, Model, Softmax, TensorWithGrad};
use crate::{Error, ModelDetails};

struct MegaManAttentionModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    multi_head_attention: MultiHeadAttention,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for MegaManAttentionModel {}

impl MegaManAttentionModel {
    pub fn new(device: &Device, sequence_length: usize, vocab_size: usize) -> Result<Self, Error> {
        // see https://github.com/karpathy/minGPT
        let _batch_size = 1;
        let n_embd = 768;
        let num_heads = 12;
        let _n_layer = 1;
        let dropout_probability = 0.01;
        let _block_size = 2048;

        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let multi_head_attention = MultiHeadAttention::try_new(
            device,
            sequence_length,
            n_embd,
            true,
            num_heads,
            dropout_probability,
        )
        .unwrap();
        let linear = Linear::new(device, vocab_size, n_embd, true, sequence_length)?;
        let softmax = Softmax::new_with_next_is_cross_entropy_loss(device);

        let model = Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            embedding,
            multi_head_attention,
            linear,
            softmax,
        };
        Ok(model)
    }
}

impl UnaryOperator for MegaManAttentionModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let embedding = self.embedding.forward(input)?;
        let attentions = self
            .multi_head_attention
            .forward(&embedding, &embedding, &embedding)?;
        let linear = self.linear.forward(&attentions)?;
        let softmax = self.softmax.forward(&linear)?;
        Ok(softmax)
    }
}

impl Model for MegaManAttentionModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}

pub fn load_mega_man_attention_model(device: &Device) -> Result<ModelDetails, Error> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let max_number_of_examples = 10;
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32;

    let input_sequence_length = sequence_length;
    let output_sequence_length = sequence_length;
    let examples = load_examples(
        &device,
        file_path,
        max_chars,
        max_number_of_examples,
        input_sequence_length,
        output_sequence_length,
        &mut tokenizer,
    )?;

    let vocab_size = tokenizer.vocab_size();
    let model = MegaManAttentionModel::new(device, sequence_length, vocab_size)?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let details = ModelDetails {
        device: device.clone(),
        tokenizer: Some(tokenizer),
        examples,
        model: Box::new(model),
        loss_operator: Box::new(loss_operator),
        optimizer: Box::new(optimizer),
        epochs: 100,
        progress: 10,
        learning_rate,
        shuffle_examples: true,
        clipped_gradient_norm: 1.0,
        initial_metrics: Metrics {
            total_loss: 1500.0,
            total_perplexity: 1600.0,
        },
        final_metrics: Metrics {
            total_loss: 100.0,
            total_perplexity: 20.0,
        },
    };
    Ok(details)
}
