use std::fs;
use std::rc::Rc;

mod architecture;
use crate::{into_one_hot_encoded_rows, Device, Operators};
use crate::{DatasetDetails, Tensor};
use architecture::*;

fn load_examples(device: &Device) -> Vec<(Tensor, Tensor)> {
    let num_classes = 256;
    let context_size = 32;
    let mut examples = Vec::new();
    let file_path = "Mega_Man.txt";
    let contents = fs::read_to_string(file_path).expect("contents");
    // TODO use bpe tokenizer.
    let tokens: Vec<usize> = contents
        .as_bytes()
        .to_owned()
        .into_iter()
        .map(|token| token as usize)
        .collect();
    println!("[load_megaman_examples] loaded {} tokens", tokens.len());
    let mut i = 0;
    let max_number_of_examples = 10;
    while i + context_size < tokens.len() && i < max_number_of_examples {
        let next_token_index = i + context_size;
        let input_tokens = &tokens[i..next_token_index];
        let one_hot_encoded_tokens = into_one_hot_encoded_rows(device, input_tokens, num_classes);
        let next_token = &tokens[next_token_index..next_token_index + 1];
        let output_multiclass = into_one_hot_encoded_rows(device, next_token, num_classes);

        examples.push((
            //
            one_hot_encoded_tokens, //
            output_multiclass,
        ));
        i += 1;
    }
    examples
}

pub fn load_dataset(device: Rc<Device>) -> DatasetDetails {
    let examples = load_examples(&device);
    let ops = Operators::new(device);
    DatasetDetails {
        examples,
        architecture: Box::new(Architecture::new(&ops)),
        epochs: 300,
        progress: 100,
        loss_function_name: ops.cross_entropy_loss(),
        initial_total_error_min: 50.0,
        final_total_error_max: 0.002,
    }
}
