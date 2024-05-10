use crate::{CrossEntropyLoss, Device, Program, Tokenizer};
use crate::{DatasetDetails, Error};
mod model;
use model::*;

use super::load_examples;

pub fn load_dataset(device: &Device) -> Result<DatasetDetails, Error> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = None;
    let max_number_of_examples = 10;
    let model = MegaManModel::new(device);
    let vocab_size = model.vocab_size();
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let input_sequence_length = model.sequence_length();
    let output_sequence_length = 1;
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
    let loss_operator = CrossEntropyLoss::new(device);
    let program = Program::try_new(&device, &model, &loss_operator)?;

    let details = DatasetDetails {
        device: device.clone(),
        tokenizer,
        examples,
        program,
        epochs: 300,
        progress: 100,
        initial_total_error_min: 50.0,
        final_total_error_max: 0.002,
        learning_rate: 0.5,
    };
    Ok(details)
}
