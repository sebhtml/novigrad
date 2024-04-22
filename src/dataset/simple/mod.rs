use std::rc::Rc;

use crate::{into_one_hot_encoded_rows, DatasetDetails, Device, Operators, Tensor};

mod architecture;
use architecture::*;

fn load_examples(device: &Device) -> Vec<(Tensor, Tensor)> {
    let mut examples = Vec::new();

    examples.push((
        //
        vec![1, 2, 3, 4, 5, 6], //
        vec![0],
    ));

    examples.push((
        //
        vec![7, 8, 9, 10, 11, 12], //
        vec![3],
    ));

    let num_classes = 16;
    let mut one_hot_encoded_input = device.tensor(0, 0, vec![]);
    let mut one_hot_encoded_output = device.tensor(0, 0, vec![]);
    let examples = examples
        .into_iter()
        .map(|example| {
            into_one_hot_encoded_rows(&example.0, num_classes, &mut one_hot_encoded_input);
            into_one_hot_encoded_rows(&example.1, num_classes, &mut one_hot_encoded_output);
            (
                one_hot_encoded_input.clone(),
                one_hot_encoded_output.clone(),
            )
        })
        .collect();

    examples
}

pub fn load_dataset(device: Rc<Device>) -> DatasetDetails {
    let examples = load_examples(&device);
    let ops = Operators::new(device);
    DatasetDetails {
        examples,
        architecture: Box::new(Architecture::new(&ops)),
        epochs: 1000,
        progress: 100,
        loss_function_name: ops.cross_entropy_loss(),
        initial_total_error_min: 4.0,
        final_total_error_max: 0.0004,
    }
}
