use crate::{
    attention_head_model::AttentionHeadModel, display::NextTokenPredictionPrinter, tensor::Error,
    Adam, Device, Metrics, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_mega_man_attention_head(
    device: &Device,
) -> Result<
    DatasetDetails<AttentionHeadModel, SoftmaxCrossEntropyLoss, Adam, NextTokenPredictionPrinter>,
    Error,
> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let max_number_of_examples = 1;
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
    let n_embd = 768;
    let causal_mask = false;
    let dropout_probability = 0.0;
    let model = AttentionHeadModel::new(
        device,
        sequence_length,
        vocab_size,
        n_embd,
        causal_mask,
        dropout_probability,
    )?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let optimizer = Adam::try_new(device, 0.05, 0.9, 0.999, 1e-8)?;
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
            total_loss: 100.0,
            total_next_token_perplexity: 5.0,
        },
        final_metrics_max: Metrics {
            total_loss: 450.0,
            total_next_token_perplexity: 16.0,
        },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size: 1,
    };
    Ok(details)
}
