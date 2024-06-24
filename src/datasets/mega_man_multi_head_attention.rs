use crate::{
    multi_head_attention_model::MultiHeadAttentionModel, neural_program::NeuralProgram,
    tensor::Error, Adam, Device, Instruction, Metrics, SoftmaxCrossEntropyLoss, Tokenizer,
    TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_mega_man_attention_dataset(
    device: &Device,
) -> Result<DatasetDetails<MultiHeadAttentionModel, SoftmaxCrossEntropyLoss, Adam>, Error> {
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
    let details = DatasetDetails {
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
    let details = load_mega_man_attention_dataset(device)?;
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