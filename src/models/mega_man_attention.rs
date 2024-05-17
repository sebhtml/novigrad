use super::load_examples;
use crate::{
    CrossEntropyLoss, Device, MultiHeadAttention, TernaryOperator, Tokenizer, UnaryModel,
    UnaryOperator,
};
use crate::{Error, ModelDetails};

use crate::{Embedding, Linear, Model, Softmax, Tensor};

struct MegaManAttentionModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    vocab_size: usize,
    sequence_length: usize,
    embedding: Embedding,
    multi_head_attention: MultiHeadAttention,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for MegaManAttentionModel {}

impl MegaManAttentionModel {
    pub fn new(device: &Device) -> Self {
        let _batch_size = 1;
        let sequence_length = 32;
        let vocab_size = 256;
        let n_embd = 384;
        let num_heads = 8;
        let _n_layer = 1;
        let _dropout = 0.1;
        let _block_size = 2048;

        let embedding = Embedding::new(device, vocab_size, n_embd);
        let multi_head_attention =
            MultiHeadAttention::try_new(device, sequence_length, n_embd, true, num_heads).unwrap();
        let linear = Linear::new(device, vocab_size, n_embd, true, sequence_length);
        let softmax = Softmax::new(device);

        Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            vocab_size,
            sequence_length,
            embedding,
            multi_head_attention,
            linear,
            softmax,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

impl UnaryOperator for MegaManAttentionModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
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
    // TODO vocab_size should be a new argument
    let model = MegaManAttentionModel::new(device);
    let vocab_size = model.vocab_size();
    let mut tokenizer = Tokenizer::ascii_tokenizer();

    let input_sequence_length = model.sequence_length();
    let output_sequence_length = input_sequence_length;
    let examples = load_examples(
        &device,
        file_path,
        max_chars,
        max_number_of_examples,
        input_sequence_length,
        output_sequence_length,
        vocab_size,
        &mut tokenizer,
    )?;

    let loss_operator = CrossEntropyLoss::new(device);

    let details = ModelDetails {
        device: device.clone(),
        tokenizer: Some(tokenizer),
        examples,
        model: Box::new(model),
        loss_operator: Box::new(loss_operator),
        epochs: 100,
        progress: 10,
        initial_total_error_min: 50.0,
        final_total_error_max: 20.0,
        learning_rate: 0.05,
        clipped_gradient_norm: 1.0,
    };
    Ok(details)
}
