use crate::{
    display::NextTokenPredictionPrinter, multi_head_attention_model::MultiHeadAttentionModel,
    neural_program::NeuralProgram, tensor::Error, Adam, Device, Instruction, Metrics,
    SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_mega_man_multi_head_attention(
    device: &Device,
) -> Result<
    DatasetDetails<
        MultiHeadAttentionModel,
        SoftmaxCrossEntropyLoss,
        Adam,
        NextTokenPredictionPrinter,
    >,
    Error,
> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let max_number_of_examples = 100;
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
    let optimizer = Adam::try_new(0.05, 0.9, 0.999, 1e-8, 0.0)?;
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 200,
        progress: 10,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics {
            total_loss: 4000.0,
            total_next_token_perplexity: 100.0,
        },
        final_metrics_max: Metrics {
            total_loss: 7000.0,
            total_next_token_perplexity: 120.0,
        },
        maximum_incorrect_predicted_next_tokens: 30,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size: 1,
    };
    Ok(details)
}

pub fn get_multi_head_attention_model_instructions(
    device: &Device,
) -> Result<Vec<Instruction>, Error> {
    let details = load_mega_man_multi_head_attention(device)?;
    let model = details.model;
    let loss_operator = details.loss_operator;
    let optimizer = details.optimizer;
    let clip_grad_norm = true;
    let program =
        NeuralProgram::try_new(device, &model, &loss_operator, &optimizer, clip_grad_norm)?;
    let instructions = program.instructions;
    Ok(instructions)
}
