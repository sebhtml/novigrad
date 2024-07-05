use crate::{
    adam_w::AdamW, display::NextTokenPredictionPrinter, tensor::Error,
    transformer_model::TransformerModel, Device, Metrics, SoftmaxCrossEntropyLoss, Tokenizer,
    TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_mega_man_transformers(
    device: &Device,
) -> Result<
    DatasetDetails<TransformerModel, SoftmaxCrossEntropyLoss, AdamW, NextTokenPredictionPrinter>,
    Error,
> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let number_of_examples = 32;
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
    let layers = 2;
    let causal_mask = true;
    let num_heads = 12;
    let dropout_probability = 0.1;
    let n_embd = 768;
    let batch_size = 32;
    let model = TransformerModel::new(
        device,
        layers,
        num_heads,
        dropout_probability,
        n_embd,
        sequence_length,
        vocab_size,
        causal_mask,
    )?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let optimizer = AdamW::try_new(0.05, 0.9, 0.999, 1e-8, 0.01)?;
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 100,
        progress: 10,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics { total_loss: 7000.0 },
        final_metrics_max: Metrics { total_loss: 150.0 },
        maximum_incorrect_predicted_next_tokens: 8,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size,
    };
    Ok(details)
}
