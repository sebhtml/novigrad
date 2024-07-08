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
    let number_of_examples = 10;
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32;

    let examples = load_examples(
        device,
        file_path,
        max_chars,
        number_of_examples,
        sequence_length,
        sequence_length,
        &mut tokenizer,
    )?;

    let vocab_size = tokenizer.vocab_size();
    let batch_size = 1;
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
        epochs: 100,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics { total_loss: 3000.0 },
        final_metrics_max: Metrics {
            total_loss: 10000.0,
        },
        maximum_incorrect_predicted_next_tokens: 10,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size,
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
    let batch_size = 1;
    let clip_grad_norm = true;
    let program = NeuralProgram::try_new(
        device,
        &model,
        &loss_operator,
        &optimizer,
        clip_grad_norm,
        batch_size,
    )?;
    let instructions = program.instructions;
    Ok(instructions)
}
