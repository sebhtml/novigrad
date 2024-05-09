use crate::{CrossEntropyLoss, Device, Tokenizer, TokenizerTrait};
use crate::{DatasetDetails, Error};
mod model;
use model::*;

use super::load_examples;

pub fn load_dataset(device: &Device) -> Result<DatasetDetails, Error> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = Some(30);
    let max_number_of_examples = 10;
    // TODO vocab_size should be a new argument
    let model = MegaManAttentionModel::new(device);
    let vocab_size = model.vocab_size();
    let mut tokenizer = Tokenizer::byte_pair_encoding();

    let input_sequence_length = model.sequence_length();
    let output_sequence_length = input_sequence_length;
    let examples = load_examples(
        &device,
        file_path,
        max_chars,
        max_number_of_examples,
        input_sequence_length,
        output_sequence_length,
        vocab_size,
        &mut tokenizer,
    )?;

    println!("TOkenizer vocab_size: {}", tokenizer.vocab_size());

    let details = DatasetDetails {
        device: device.clone(),
        tokenizer,
        examples,
        model: Box::new(model),
        epochs: 1000,
        progress: 100,
        loss_function_name: Box::new(CrossEntropyLoss::new(device)),
        initial_total_error_min: 50.0,
        final_total_error_max: 0.002,
        learning_rate: 0.5,
    };
    Ok(details)
}
