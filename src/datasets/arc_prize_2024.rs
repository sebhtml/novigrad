use core::panic;
use std::fs;

use serde::{Deserialize, Serialize};

use crate::{
    adam_w::AdamW,
    display::BoardPrinter,
    error,
    tensor::{Error, ErrorEnum},
    transformer_model::TransformerModel,
    Device, Metrics, SoftmaxCrossEntropyLoss, TensorWithGrad,
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

fn load_json_examples(
    training_or_evaluation: &str,
    problem_id: &str,
    train_or_test: &str,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>, Error> {
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

    Ok(examples)
}

fn put_pixels_on_larger_board(
    pixels: Vec<usize>,
    old_width: usize,
    old_height: usize,
    new_width: usize,
    new_height: usize,
    default_pixel: usize,
) -> Vec<usize> {
    let mut new_pixels = vec![default_pixel; new_width * new_height];
    let x_translation = (new_width - old_width) / 2;
    let y_translation = (new_height - old_height) / 2;
    for old_x in 0..old_height {
        for old_y in 0..old_width {
            let pixel = pixels[old_y * old_width + old_x];
            let new_x = old_x + x_translation;
            let new_y = old_y + y_translation;
            new_pixels[new_y * new_width + new_x] = pixel;
        }
    }
    new_pixels
}

fn put_example_on_larger_board(
    examples: Vec<(Vec<usize>, Vec<usize>)>,
    old_width: usize,
    old_height: usize,
    new_width: usize,
    new_height: usize,
    default_pixel: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let examples = examples
        .into_iter()
        .map(|(input, output)| {
            let input = put_pixels_on_larger_board(
                input,
                old_width,
                old_height,
                new_width,
                new_height,
                default_pixel,
            );
            let output = put_pixels_on_larger_board(
                output,
                old_width,
                old_height,
                new_width,
                new_height,
                default_pixel,
            );
            (input, output)
        })
        .collect::<Vec<_>>();
    examples
}

fn load_examples(
    training_or_evaluation: &str,
    problem_id: &str,
    train_or_test: &str,
    device: &Device,
    vocab_size: usize,
    new_width: usize,
    new_height: usize,
    default_pixel: usize,
) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let examples = load_json_examples(training_or_evaluation, problem_id, train_or_test)?;
    let old_width = (examples[0].0.len() as f32).sqrt() as usize;
    let old_height = old_width;
    let examples = put_example_on_larger_board(
        examples,
        old_width,
        old_height,
        new_width,
        new_height,
        default_pixel,
    );
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
) -> Result<DatasetDetails<TransformerModel, SoftmaxCrossEntropyLoss, AdamW, BoardPrinter>, Error> {
    let vocab_size = 10;
    let width = 21;
    let height = 21;
    let default_pixel = 0;
    let sequence_length = width * height;
    let training_examples = load_examples(
        "training",
        "3aa6fb7a",
        "train",
        device,
        vocab_size,
        width,
        height,
        default_pixel,
    )?;
    let test_examples = load_examples(
        "training",
        "3aa6fb7a",
        "test",
        device,
        vocab_size,
        width,
        height,
        default_pixel,
    )?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let optimizer = AdamW::try_new(0.05, 0.9, 0.999, 1e-8, 0.1)?;
    let layers = 2;
    let num_heads = 16;
    let dropout_probability = 0.1;
    let n_embd = 2048;
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
        epochs: 50,
        progress: 10,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics { total_loss: 5.0 },
        final_metrics_max: Metrics { total_loss: 0.0 },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: BoardPrinter::new(width, height),
        batch_size: 1,
    };
    Ok(details)
}
