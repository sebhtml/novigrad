use crate::{
    display::NextTokenPredictionPrinter,
    error,
    simple::SimpleModel,
    stochastic_gradient_descent::StochasticGradientDescent,
    tensor::{Error, ErrorEnum},
    Device, Metrics, SoftmaxCrossEntropyLoss, TensorWithGrad, Tokenizer, TokenizerTrait,
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

pub fn load_simple(
    device: &Device,
) -> Result<
    DatasetDetails<
        SimpleModel,
        SoftmaxCrossEntropyLoss,
        StochasticGradientDescent,
        NextTokenPredictionPrinter,
    >,
    Error,
> {
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 6;
    let examples = load_examples(device, &mut tokenizer)?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let vocab_size = tokenizer.vocab_size();
    let model = SimpleModel::new(device, sequence_length, vocab_size)?;
    let optimizer = StochasticGradientDescent::new(0.5);
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 500,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics { total_loss: 5.0 },
        final_metrics_max: Metrics { total_loss: 1e-4 },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size: 1,
    };
    Ok(details)
}
