use super::load_examples;
use crate::{
    CrossEntropyLoss, Device, MultiHeadAttention, NeuralMachine, TernaryOperator, Tokenizer,
    TokenizerTrait, UnaryOperator,
};
use crate::{DatasetDetails, Error};

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

impl MegaManAttentionModel {
    pub fn new(device: &Device) -> Self {
        let _batch_size = 1;
        let sequence_length = 6;
        let vocab_size = 20;
        let n_embd = 64; // 384; needs LayerNorm
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

pub fn load_dataset(device: &Device) -> Result<DatasetDetails, Error> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = Some(30);
    let max_number_of_examples = 10;
    // TODO vocab_size should be a new argument
    let model = MegaManAttentionModel::new(device);
    let vocab_size = model.vocab_size();
    let mut tokenizer = Tokenizer::byte_pair_encoding();

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

    println!("TOkenizer vocab_size: {}", tokenizer.vocab_size());

    let model = model;
    let loss_operator = CrossEntropyLoss::new(device);
    let program = NeuralMachine::try_new(&device, &model, &loss_operator)?;

    let details = DatasetDetails {
        device: device.clone(),
        tokenizer: Some(tokenizer),
        examples,
        program,
        epochs: 1000,
        progress: 100,
        initial_total_error_min: 25.0,
        final_total_error_max: 100.0, // The loss may be bad but the next token prediction is good and it's tested separately
        learning_rate: 0.5,
    };
    Ok(details)
}
