use crate::{
    display::NextTokenPredictionPrinter, mega_man::MegaManModel, tensor::Error, Device,
    GradientDescent, Metrics, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_mega_man_linear_dataset(
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
    let max_number_of_examples = 10;
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
        clipped_gradient_norm: true,
        initial_metrics: Metrics {
            total_loss: 50.0,
            total_next_token_perplexity: 2500.0,
        },
        final_metrics: Metrics {
            total_loss: 0.0,
            total_next_token_perplexity: 11.0,
        },
        maximum_incorrect_argmaxes: 0,
        printer: NextTokenPredictionPrinter::new(tokenizer),
    };
    Ok(details)
}
