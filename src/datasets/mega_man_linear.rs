use crate::{
    display::NextTokenPredictionPrinter, mega_man::MegaManModel, tensor::Error, Device,
    GradientDescent, Metrics, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_mega_man_linear(
    device: &Device,
) -> Result<
    DatasetDetails<
        MegaManModel,
        SoftmaxCrossEntropyLoss,
        GradientDescent,
        NextTokenPredictionPrinter,
    >,
    Error,
> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let max_number_of_examples = 1000;
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32;
    let input_sequence_length = sequence_length;
    let output_sequence_length = 1;
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
    let model = MegaManModel::new(device, sequence_length, vocab_size)?;
    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.5;
    let optimizer = GradientDescent::new(learning_rate);
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 100,
        progress: 10,
        learning_rate,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics {
            total_loss: 5500.0,
            total_next_token_perplexity: 250000.0,
        },
        final_metrics_max: Metrics {
            total_loss: 40.0,
            total_next_token_perplexity: 1010.0,
        },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size: 64,
    };
    Ok(details)
}
