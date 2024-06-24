use crate::datasets::load_examples;
use crate::neural_program::NeuralProgram;
use crate::{tensor::Error, ModelDetails};
use crate::{
    Adam, Device, Instruction, Metrics, MultiHeadAttention, SoftmaxCrossEntropyLoss,
    TernaryOperator, Tokenizer, TokenizerTrait, UnaryModel, UnaryOperator, WeightsInitialization,
};
use crate::{Embedding, Linear, Model, Softmax, TensorWithGrad};

pub struct MultiHeadAttentionModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    multi_head_attention: MultiHeadAttention,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for MultiHeadAttentionModel {}

impl MultiHeadAttentionModel {
    pub fn new(device: &Device, sequence_length: usize, vocab_size: usize) -> Result<Self, Error> {
        let n_embd = 768;
        let num_heads = 12;
        let dropout_probability = 0.0;

        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let causal_mask = true;
        let multi_head_attention = MultiHeadAttention::try_new(
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
            multi_head_attention,
            linear,
            softmax,
        };
        Ok(model)
    }
}

impl UnaryOperator for MultiHeadAttentionModel {
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

impl Model for MultiHeadAttentionModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}

pub fn load_multi_head_attention_model(
    device: &Device,
) -> Result<ModelDetails<MultiHeadAttentionModel, SoftmaxCrossEntropyLoss, Adam>, Error> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let max_number_of_examples = 10;
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32;

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
    let model = MultiHeadAttentionModel::new(device, sequence_length, vocab_size)?;

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
        clipped_gradient_norm: true,
        initial_metrics: Metrics {
            total_loss: 4000.0,
            total_perplexity: 5.0,
        },
        final_metrics: Metrics {
            total_loss: 350.0,
            total_perplexity: 13.0,
        },
        maximum_incorrect_argmaxes: 2,
    };
    Ok(details)
}

pub fn get_multi_head_attention_model_instructions(
    device: &Device,
) -> Result<Vec<Instruction>, Error> {
    let details = load_multi_head_attention_model(device)?;
    let model = details.model;
    let loss_operator = details.loss_operator;
    let optimizer = details.optimizer;
    let clipped_gradient_norm = true;
    let program = NeuralProgram::try_new(
        device,
        &model,
        &loss_operator,
        &optimizer,
        clipped_gradient_norm,
    )?;
    let instructions = program.instructions;
    Ok(instructions)
}
