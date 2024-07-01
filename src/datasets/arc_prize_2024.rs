use core::panic;
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
    pub test: Vec<Example>,
}

#[derive(Serialize, Deserialize)]
struct Example {
    pub input: Vec<Vec<usize>>,
    pub output: Vec<Vec<usize>>,
}

fn load_examples(
    training_or_evaluation: &str,
    problem_id: &str,
    train_or_test: &str,
    device: &Device,
    vocab_size: usize,
) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let file_path =
        format!("/home/sebhtml/projects/ARC-AGI/data/{training_or_evaluation}/{problem_id}.json");
    let data = fs::read_to_string(file_path).unwrap();
    let p: Problem = serde_json::from_str(&data).unwrap();
    let examples = match train_or_test {
        "train" => p.train,
        "test" => p.test,
        _ => panic!(),
    };
    let examples = examples
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

pub fn load_arc_prize_2024(
    device: &Device,
) -> Result<DatasetDetails<TransformerModel, SoftmaxCrossEntropyLoss, Adam, BoardPrinter>, Error> {
    let vocab_size = 10;
    let sequence_length = 7 * 7;
    let training_examples = load_examples("training", "3aa6fb7a", "train", device, vocab_size)?;
    let test_examples = load_examples("training", "3aa6fb7a", "test", device, vocab_size)?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let layers = 2;
    let num_heads = 12;
    let dropout_probability = 0.1;
    let n_embd = 768;
    let causal_mask = false;
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
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: training_examples,
        test_examples,
        model,
        loss_operator,
        optimizer,
        epochs: 500,
        progress: 100,
        learning_rate,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics {
            total_loss: 5.0,
            total_next_token_perplexity: 200.0,
        },
        final_metrics_max: Metrics {
            total_loss: 0.0,
            total_next_token_perplexity: 2.0,
        },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: BoardPrinter::default(),
        batch_size: 1,
    };
    Ok(details)
}
