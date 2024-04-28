use std::rc::Rc;

use crate::{into_one_hot_encoded_rows, DatasetDetails, Device, Error, LearningTensor, Operators};

mod architecture;
use architecture::*;

fn load_examples(device: &Device) -> Result<Vec<(LearningTensor, LearningTensor)>, Error> {
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
    let examples = examples
        .into_iter()
        .map(|example| {
            let one_hot_encoded_input = into_one_hot_encoded_rows(device, &example.0, num_classes);
            let one_hot_encoded_output = into_one_hot_encoded_rows(device, &example.1, num_classes);
            (one_hot_encoded_input, one_hot_encoded_output)
        })
        .try_fold(vec![], |mut acc, item| match item {
            (Ok(a), Ok(b))   => {
                acc.push((a, b));
                Ok(acc)
            }
            _ => Err(Error::UnsupportedOperation)
        });

    examples
}

pub fn load_dataset(device: Rc<Device>) -> Result<DatasetDetails, Error> {
    let examples = load_examples(&device)?;
    let ops = Operators::new(device);
    let details = DatasetDetails {
        examples,
        architecture: Box::new(Architecture::new(&ops)),
        epochs: 1000,
        progress: 100,
        loss_function_name: ops.cross_entropy_loss(),
        initial_total_error_min: 4.0,
        final_total_error_max: 0.0005,
    };
    Ok(details)
}
