use std::ops::Deref;

use crate::{DatasetDetails, Error, Network, Tensor, TensorF32, Tokenizer, TokenizerTrait};

pub fn print_expected_output_and_actual_output(
    tokenizer: &mut Tokenizer,
    example: usize,
    input: &TensorF32,
    expected_output: &TensorF32,
    actual_output: &TensorF32,
    expected_argmax: usize,
    actual_argmax: usize,
    loss: f32,
) -> Result<(), Error> {
    let rows = expected_output.rows();
    let cols = expected_output.cols();

    let input_tokens = get_row_argmaxes(input)?;
    let expected_output_token = [expected_argmax];
    let actual_output_token = [actual_argmax];

    println!("----");
    println!("Example {}", example);
    println!("Loss {}", loss);

    println!("  input_tokens: {:?}", &input_tokens);
    println!("  input_text: {}", tokenizer.decode(&input_tokens)?);

    println!("  expected_output_token: {:?}", &expected_output_token);
    println!(
        "  expected_output_text: {}",
        tokenizer.decode(&expected_output_token)?
    );

    println!("  actual_output_token: {:?}", &actual_output_token);
    println!(
        "  actual_output_text: {}",
        tokenizer.decode(&actual_output_token)?
    );

    let expected_values = expected_output.get_values()?;
    let actual_values = actual_output.get_values()?;
    println!("");

    for row in 0..rows {
        for col in 0..cols {
            println!(
                "index {}  expected {}  actual {}",
                col,
                expected_values[expected_output.index(row, col)],
                actual_values[actual_output.index(row, col)],
            );
        }
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
    let total_error = network.total_loss(inputs, outputs)?;
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
    let examples = &dataset_details.examples;
    let learning_rate = dataset_details.learning_rate;
    let architecture = dataset_details.model;
    let loss_function = dataset_details.loss_function_name;
    let mut tokenizer = dataset_details.tokenizer;
    let device = dataset_details.device;

    let inputs: Vec<_> = examples.iter().map(|x| x.clone().0).collect();
    let outputs: Vec<_> = examples.iter().map(|x| x.clone().1).collect();
    let mut network = Network::new(architecture, loss_function, &device);

    let mut last_total_error = f32::NAN;
    let epochs = dataset_details.epochs;
    let progress = dataset_details.progress;

    let (_, _) = print_results(&mut tokenizer, &mut network, &inputs, &outputs)?;

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
        network.train(learning_rate, epoch, &inputs, &outputs)?;
    }
    let final_total_error =
        print_total_error(&mut network, &inputs, &outputs, last_total_error, epochs)?;

    let (expected_argmax_values, actual_argmax_values) =
        print_results(&mut tokenizer, &mut network, &inputs, &outputs)?;

    let output = NetworkTestOutput {
        initial_total_error,
        final_total_error,
        expected_argmax_values,
        actual_argmax_values,
    };
    Ok(output)
}

fn print_results(
    tokenizer: &mut Tokenizer,
    network: &mut Network,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let mut expected_argmax_values = Vec::new();
    let mut actual_argmax_values = Vec::new();

    for i in 0..inputs.len() {
        let input = &inputs[i];
        let actual_output = network.model().forward(&[input.clone()])?;
        let loss = network.example_loss(&actual_output, &outputs[i])?;

        let expected_output: &TensorF32 = &outputs[i].tensor().deref().borrow();
        let expected_output_argmaxes = get_row_argmaxes(expected_output)?;
        let expected_argmax = expected_output_argmaxes[0];
        expected_argmax_values.push(expected_argmax);

        let actual_output: &TensorF32 = &actual_output.tensor().deref().borrow();
        let actual_output_argmaxes = get_row_argmaxes(actual_output)?;
        let actual_argmax = actual_output_argmaxes[0];
        actual_argmax_values.push(actual_argmax);

        print_expected_output_and_actual_output(
            tokenizer,
            i,
            &input.tensor().deref().borrow(),
            expected_output,
            actual_output,
            expected_argmax,
            actual_argmax,
            loss,
        )?;
    }

    Ok((expected_argmax_values, actual_argmax_values))
}

fn get_row_argmaxes(tensor: &TensorF32) -> Result<Vec<usize>, Error> {
    let values = tensor.get_values()?;
    let cols = tensor.cols();
    let mut argmaxes = vec![];
    for row in 0..tensor.rows() {
        let mut argmax_col = 0;
        for col in 0..cols {
            let acc = values[tensor.index(row, argmax_col)];
            let item = values[tensor.index(row, col)];
            if item > acc {
                argmax_col = col;
            }
        }
        argmaxes.push(argmax_col);
    }
    Ok(argmaxes)
}
