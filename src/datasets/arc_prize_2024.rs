use std::fs;

use serde::{Deserialize, Serialize};

use crate::{
    display::BoardPrinter,
    error,
    tensor::{Error, ErrorEnum},
    transformer_model::TransformerModel,
    Adam, Device, Metrics, SoftmaxCrossEntropyLoss, TensorWithGrad,
};

use super::{into_one_hot_encoded_rows, DatasetDetails};

#[derive(Serialize, Deserialize)]
struct Problem {
    pub train: Vec<Example>,
}

#[derive(Serialize, Deserialize)]
struct Example {
    pub input: Vec<Vec<usize>>,
    pub output: Vec<Vec<usize>>,
}

fn load_examples(
    device: &Device,
    vocab_size: usize,
) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let file_path = "/home/sebhtml/projects/ARC-AGI/data/training/3aa6fb7a.json";
    let data = fs::read_to_string(file_path).unwrap();
    let p: Problem = serde_json::from_str(&data).unwrap();
    let examples = p
        .train
        .iter()
        .map(|e| {
            let input = e.input.concat();
            let output = e.output.concat();
            (input, output)
        })
        .collect::<Vec<_>>();
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

pub fn load_arc_dataset(
    device: &Device,
) -> Result<DatasetDetails<TransformerModel, SoftmaxCrossEntropyLoss, Adam, BoardPrinter>, Error> {
    let vocab_size = 10;
    let sequence_length = 7 * 7;
    let examples = load_examples(device, vocab_size)?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let layers = 1;
    let causal_mask = false;
    let model = TransformerModel::new(device, layers, sequence_length, vocab_size, causal_mask)?;
    let details = DatasetDetails {
        device: device.clone(),
        examples,
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
            total_perplexity: 200.0,
        },
        final_metrics: Metrics {
            total_loss: 0.0,
            total_perplexity: 2.0,
        },
        maximum_incorrect_argmaxes: 0,
        printer: BoardPrinter::default(),
    };
    Ok(details)
}
