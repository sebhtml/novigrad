use crate::{
    display::NextTokenPredictionPrinter, mega_man::MegaManModel,
    stochastic_gradient_descent::StochasticGradientDescent, tensor::Error, Device, Metrics,
    SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_mega_man_linear(
    device: &Device,
) -> Result<
    DatasetDetails<
        MegaManModel,
        SoftmaxCrossEntropyLoss,
        StochasticGradientDescent,
        NextTokenPredictionPrinter,
    >,
    Error,
> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let number_of_examples = 1024;
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32;
    let output_sequence_length = 1;
    let examples = load_examples(
        device,
        file_path,
        max_chars,
        number_of_examples,
        sequence_length,
        output_sequence_length,
        &mut tokenizer,
    )?;
    let vocab_size = tokenizer.vocab_size();
    let batch_size = 4;
    let model = MegaManModel::new(device, sequence_length, vocab_size)?;
    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let optimizer = StochasticGradientDescent::new(0.5);
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 50,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics { total_loss: 5500.0 },
        final_metrics_max: Metrics { total_loss: 0.01 },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size,
    };
    Ok(details)
}
