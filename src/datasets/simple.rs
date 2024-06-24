use crate::{
    display::NextTokenPredictionPrinter,
    error,
    simple::SimpleModel,
    tensor::{Error, ErrorEnum},
    Device, GradientDescent, Metrics, SoftmaxCrossEntropyLoss, TensorWithGrad, Tokenizer,
    TokenizerTrait,
};

use super::{into_one_hot_encoded_rows, DatasetDetails};

fn load_examples(
    device: &Device,
    tokenizer: &mut Tokenizer,
) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let examples: Vec<_> = ["quizzed", "fuzzing"]
        .iter()
        .map(|text| {
            (
                tokenizer.encode(&text[0..text.len() - 1]),
                tokenizer.encode(&text[text.len() - 1..text.len()]),
            )
        })
        .collect();

    let vocab_size = tokenizer.vocab_size();

    examples
        .into_iter()
        .map(|example| {
            let one_hot_encoded_input = into_one_hot_encoded_rows(device, &example.0, vocab_size);
            let one_hot_encoded_output = into_one_hot_encoded_rows(device, &example.1, vocab_size);
            (one_hot_encoded_input, one_hot_encoded_output)
        })
        .try_fold(vec![], |mut acc, item| match item {
            (Ok(a), Ok(b)) => {
                acc.push((a, b));
                Ok(acc)
            }
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        })
}

pub fn load_simple_dataset(
    device: &Device,
) -> Result<
    DatasetDetails<
        SimpleModel,
        SoftmaxCrossEntropyLoss,
        GradientDescent,
        NextTokenPredictionPrinter,
    >,
    Error,
> {
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 6;
    let examples = load_examples(device, &mut tokenizer)?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.5;
    let vocab_size = tokenizer.vocab_size();
    let model = SimpleModel::new(device, sequence_length, vocab_size)?;
    let optimizer = GradientDescent::new(learning_rate);
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 500,
        progress: 100,
        learning_rate,
        shuffle_examples: true,
        clipped_gradient_norm: true,
        initial_metrics: Metrics {
            total_loss: 5.0,
            total_next_token_perplexity: 200.0,
        },
        final_metrics: Metrics {
            total_loss: 0.0,
            total_next_token_perplexity: 2.0,
        },
        maximum_incorrect_argmaxes: 0,
        printer: NextTokenPredictionPrinter::new(tokenizer),
    };
    Ok(details)
}
