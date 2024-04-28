use std::ops::Deref;

use crate::{DatasetDetails, Error, Network, Tensor, TensorF32};

pub fn print_expected_output_and_actual_output(
    example: usize,
    expected_output: &TensorF32,
    actual_output: &TensorF32,
    expected_argmax: usize,
    actual_argmax: usize,
    loss: Option<&TensorF32>,
) -> Result<(), Error> {
    let cols = expected_output.cols();
    let last_row = actual_output.rows() - 1;

    println!("----");
    println!("Example {}", example);
    println!(
        "expected_argmax {}, actual_argmax {}",
        expected_argmax, actual_argmax
    );

    let expected_values = expected_output.get_values()?;
    let actual_values = actual_output.get_values()?;
    println!("");
    for col in 0..cols {
        // TODO is this the correct loss in the loss tensor
        let loss = match loss {
            Some(loss) => {
                let values = loss.get_values()?;
                values[loss.index(0, col)]
            }
            _ => Default::default(),
        };
        println!(
            "index {}  expected {}  actual {}  loss {}",
            col,
            // TODO last_row should not be a thing
            expected_values[expected_output.index(0, col)],
            actual_values[actual_output.index(last_row, col)],
            loss
        );
    }
    Ok(())
}

fn print_total_error(
    network: &mut Network,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
    last_total_error: f32,
    epoch: usize,
) -> Result<f32, Error> {
    let total_error = network.total_error(inputs, outputs)?;
    let change = (total_error - last_total_error) / last_total_error;
    println!(
        "Epoch {} Total_error {}, change: {}",
        epoch, total_error, change
    );
    Ok(total_error)
}

pub struct NetworkTestOutput {
    pub initial_total_error: f32,
    pub final_total_error: f32,
    pub expected_argmax_values: Vec<usize>,
    pub actual_argmax_values: Vec<usize>,
}

pub fn train_network_on_dataset(
    dataset_details: DatasetDetails,
) -> Result<NetworkTestOutput, Error> {
    let mut initial_total_error = f32::NAN;
    let examples = dataset_details.examples;
    let architecture = dataset_details.architecture;
    let loss_function_name = dataset_details.loss_function_name;

    let inputs = examples.iter().map(|x| x.clone().0).collect();
    let outputs = examples.iter().map(|x| x.clone().1).collect();
    let mut network = Network::new(architecture, loss_function_name);

    let mut last_total_error = f32::NAN;
    let epochs = dataset_details.epochs;
    let progress = dataset_details.progress;

    let (_, _) = print_results(&mut network, &inputs, &outputs)?;

    for epoch in 0..epochs {
        if epoch % progress == 0 {
            let total_error =
                print_total_error(&mut network, &inputs, &outputs, last_total_error, epoch)?;
            if epoch == 0 {
                initial_total_error = total_error;
            }
            last_total_error = total_error;
            if last_total_error == 0.0 {
                break;
            }
        }
        network.train(epoch, &inputs, &outputs)?;
    }
    let final_total_error =
        print_total_error(&mut network, &inputs, &outputs, last_total_error, epochs)?;

    let (expected_argmax_values, actual_argmax_values) =
        print_results(&mut network, &inputs, &outputs)?;

    let output = NetworkTestOutput {
        initial_total_error,
        final_total_error,
        expected_argmax_values,
        actual_argmax_values,
    };
    Ok(output)
}

fn print_results(
    network: &mut Network,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let activation_tensors = network.predict_many(&inputs).unwrap();

    let mut expected_argmax_values = Vec::new();
    let mut actual_argmax_values = Vec::new();

    for i in 0..inputs.len() {
        let expected_output: &TensorF32 = &outputs[i].tensor().deref().borrow();
        let actual_output: &TensorF32 = &activation_tensors[i].tensor().deref().borrow();

        let expected_values = expected_output.get_values()?;
        let actual_values = actual_output.get_values()?;
        let cols = expected_output.cols();
        let mut expected_argmax = 0;
        for col in 0..cols {
            if expected_values[expected_output.index(0, col)]
                > expected_values[expected_output.index(0, expected_argmax)]
            {
                expected_argmax = col;
            }
        }

        let last_row = actual_output.rows() - 1;
        let mut actual_argmax = 0;
        for col in 0..cols {
            if actual_values[actual_output.index(last_row, col)]
                > actual_values[actual_output.index(last_row, actual_argmax)]
            {
                actual_argmax = col;
            }
        }

        expected_argmax_values.push(expected_argmax);
        actual_argmax_values.push(actual_argmax);

        print_expected_output_and_actual_output(
            i,
            expected_output,
            actual_output,
            expected_argmax,
            actual_argmax,
            None,
        )?;
    }

    // TODO add a separate function that compute argmaxes instead of computing them in here.
    Ok((expected_argmax_values, actual_argmax_values))
}
