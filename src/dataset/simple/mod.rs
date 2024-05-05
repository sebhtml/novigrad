use crate::{
    into_one_hot_encoded_rows, DatasetDetails, Device, Error, Operators, Tensor, Tokenizer,
    TokenizerTrait,
};

mod model;
use model::*;

fn load_examples(
    device: &Device,
    tokenizer: &mut Tokenizer,
) -> Result<Vec<(Tensor, Tensor)>, Error> {
    let examples: Vec<_> = ["quizzed", "fuzzing"]
        .iter()
        .map(|text| {
            (
                tokenizer.encode(&text[0..text.len() - 1]),
                tokenizer.encode(&text[text.len() - 1..text.len()]),
            )
        })
        .collect();

    let num_classes = 256;
    let examples = examples
        .into_iter()
        .map(|example| {
            let one_hot_encoded_input = into_one_hot_encoded_rows(device, &example.0, num_classes);
            let one_hot_encoded_output = into_one_hot_encoded_rows(device, &example.1, num_classes);
            (one_hot_encoded_input, one_hot_encoded_output)
        })
        .try_fold(vec![], |mut acc, item| match item {
            (Ok(a), Ok(b)) => {
                acc.push((a, b));
                Ok(acc)
            }
            _ => Err(Error::UnsupportedOperation),
        });

    examples
}

pub fn load_dataset(device: &Device) -> Result<DatasetDetails, Error> {
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let examples = load_examples(&device, &mut tokenizer)?;
    let ops = Operators::new(device);
    let details = DatasetDetails {
        device: device.clone(),
        tokenizer,
        examples,
        model: Box::new(Model::new(&ops)),
        epochs: 1000,
        progress: 100,
        loss_function_name: Box::new(ops.cross_entropy_loss()),
        initial_total_error_min: 8.0,
        final_total_error_max: 0.001,
        learning_rate: 0.5,
    };
    Ok(details)
}
